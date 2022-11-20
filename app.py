import rasterio
import rasterio.warp
from rasterio.crs import CRS
from area import area
import uuid
from flask import Flask, render_template, request, url_for, flash, redirect, session
from flask_login import LoginManager, UserMixin
from flask_session import Session
from PIL import Image
import numpy as np
import cv2 as cv
from processing import make_pred_good
from processing import placeMaskOnImg
import os
import glob
import segmentation_models as sm
from keras import backend as K
from tensorflow import keras
import tensorflow as tf
import sqlite3
import bcrypt
from datetime import date
from datetime import datetime
import mail

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.config.experimental.set_visible_devices([], 'GPU')

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0) 
 
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

objects = {
        'dice_coef' : dice_coef,
        'dice_coef_loss' : dice_coef_loss,
        'iou_score' : sm.metrics.iou_score
        }
model = keras.models.load_model('projeto-hu-unet3-equalized-b6-e100-2.h5', custom_objects = objects)

app = Flask(__name__, static_url_path='/static')
app.secret_key = "super secret key"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

def get_db_connection():
    conn = sqlite3.connect('gardens.db')
    conn.row_factory = sqlite3.Row
    return conn

class User(UserMixin):
    def __init__(self, id, name, email, password, password_token):
        self.id = id
        self.name = name
        self.email = email
        self.password = password
        self.password_token = password_token
        self.authenticated = False
    def is_active(self):
         return self.is_active()
    def is_anonymous(self):
         return False
    def is_authenticated(self):
         return self.authenticated
    def is_active(self):
         return True
    def get_id(self):
         return self.id

@login_manager.user_loader
def load_user(user_id):
   conn = get_db_connection()
   curs = conn.cursor()
   curs.execute("SELECT * from user where id = (?)", [user_id])
   lu = curs.fetchone()
   print(lu)
   if lu is None:
        return None
   else:
        return User(lu[0], lu[1], lu[2], lu[3], lu[4])
    
@app.route("/")
def home():
    return redirect(url_for('login'))

@app.route("/login", methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        print(email, password)
        if not email:
            flash('Preencha seu email.')
        elif not password:
            flash('Preencha sua senha.')
        else:
            conn = get_db_connection()
            curs = conn.cursor()
            curs.execute('SELECT * FROM user where email = (?)', [email]) 
            user = list(curs.fetchone())
            print(user[0])
            Us = load_user(user[0])
            
            # un-hash password
            password = str.encode(password)
            hashed = str.encode(str(Us.password))
            ##########################
            
            #if (str(email) == str(Us.email) and str(password) == str(Us.password)):
            if (str(email) == str(Us.email) and bcrypt.checkpw(password, hashed)):
                session['id'] = [str(Us.id)]
                return redirect(url_for('index', id=Us.id))
            else:
                flash('E-mail ou senha incorretos.')
    return render_template('login.html')

@app.route('/logout')
def logout():
   # remove the username from the session if it is there
   session.pop('id', None)
   return redirect(url_for('login'))

@app.route("/get_token", methods=('GET', 'POST'))
def get_token():
    if request.method == 'POST':
        if request.form['button'] == "Recuperar senha":
            email = request.form['email']
            conn = get_db_connection()
            users = conn.execute("SELECT * FROM user where email = (?)", [email]).fetchall()
        
            if len(users)>0:
                code = str(uuid.uuid4())
                conn = get_db_connection()
                conn.execute('UPDATE user SET password_token = (?) WHERE email = (?)', (code, email))
                conn.commit()
                conn.close()
                mail.send_mail(email, code)
                flash('Novo token enviado.')
            else:
                flash('E-mail não encontrado.')
                       
    return render_template('get_token.html')
    
@app.route("/insert_token", methods=('GET', 'POST'))
def insert_token():
    if request.method == 'POST':
        if request.form['button'] == "Redefinir senha":
            token = request.form['token']
            new_password = request.form['new_password']
            password = request.form['password']
            
            if new_password != password:
                flash('Senhas não combinam.')
            else:
                conn = get_db_connection()
                curs = conn.cursor()
                user = curs.execute('SELECT * FROM user where password_token = (?)', [token]).fetchone()
                if user != None:
                    user = list(user)
                    Us = load_user(user[0])
                    
                    # hash password
                    password = str.encode(password)
                    hashed = bcrypt.hashpw(password, bcrypt.gensalt())
                    hashed = hashed.decode('utf-8')
                    ################
                    
                    conn.execute('UPDATE user SET password = (?) WHERE email = (?)', (hashed, Us.email))
                    conn.execute('UPDATE user SET password_token = (?) WHERE email = (?)', ("", Us.email))
                    conn.commit()
                    conn.close()
                    Us = ""
                    return redirect(url_for('login'))
                else:
                    flash('Token inválido!')
                    
           
    return render_template('insert_token.html')

    

@app.route("/registration", methods=('GET', 'POST'))
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        accepted_terms = request.form['accepted_terms']
        
        # checar se email cadastrado já existe
        conn = get_db_connection()
        users = conn.execute("SELECT * FROM user where email = (?)", [email]).fetchall()
        conn.close()
                
        if len(users) > 0:
            flash('Email já cadastrado!')
            return render_template('registration.html')
                
        # hash password
        password = str.encode(password)
        hashed = bcrypt.hashpw(password, bcrypt.gensalt())
        hashed = hashed.decode('utf-8')
        ########################################
        
        if not name:
           pass #flash('Name is required!')
        elif not email:
           pass #flash('Email is required!')
        elif not password:
           pass # flash('Password is required!')
        else:
            conn = get_db_connection()
            conn.execute('INSERT INTO user (name, email, password, accepted_terms) VALUES (?, ?, ?, ?)',
                         (name, email, hashed, accepted_terms))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))

    return render_template('registration.html')

@app.route("/home2", methods=('GET', 'POST'))
def home2():
    session_id = session['id'][0]
    return redirect(url_for('index', id=session_id))
    
@app.route("/index/<id>", methods=('GET', 'POST'))
def index(id):
    
    if session.get("id"):
        session_id = session['id'][0]
    else:
        session_id = 0
        
    if id == session_id: 
        conn = get_db_connection()
        posts = conn.execute('SELECT * FROM image where user_id = (?)', [id]).fetchall()
        conn.close()
        if request.method == 'GET':
            return render_template('index.html', posts=posts)
        
        if request.method == 'POST':
            if request.form['button'] == "Carregar":
                file = request.files['file']
                description = request.form['description']
                
                if not file or not description:
                    return render_template('index.html', posts=posts)
                
                if file:
                    file_name = str(file.filename)
                    extension = file_name.split('.')[-1]
                    
                    if (extension == 'tif' or extension == "tiff"):
                        image_path = "static/images/" + file.filename
                        file.save(image_path)
                
                        dat = rasterio.open(image_path)
                        bounds = dat.bounds
                        
                        x_min, y_min, x_max, y_max = bounds
                        feature = {
                            "type": "Polygon",
                            "coordinates": [
                            [[x_max, y_min], 
                                [x_max, y_max], 
                                [x_min, y_max], 
                                [x_min, y_min], 
                                [x_max, y_min]]
                            ]
                        }

                        feature_proj = rasterio.warp.transform_geom(
                            dat.crs,
                            CRS.from_epsg(4326),
                            feature
                        )
                    
                        #poligono e area
                        polygon = str(feature_proj['coordinates'])
                        obj = feature_proj
                        area_m2 = str(area(obj))
                        #####
                        
                        data = dat.read([1,2,3])
                        def normalize(x, lower, upper):
                            x_max = np.max(x)
                            x_min = np.min(x)

                            m = (upper - lower) / (x_max - x_min)
                            x_norm = (m * (x - x_min)) + lower

                            return x_norm

                        # Normalize each band separately
                        data_norm = np.array([normalize(data[i,:,:], 0, 255) for i in range(data.shape[0])])
                        im = Image.fromarray(np.moveaxis(data, 0, -1))
                        image_path = "static/images/"+file_name.split('.')[0]+".jpg"
                        im.save(image_path)                        
                        print('salvo como jpg')
                        
                    elif (extension == 'jpeg' or extension == "jpg" or extension == "png"):
                        polygon = ''
                        area_m2 = ''
                        image_path = "static/images/" + file.filename
                        file.save(image_path)
                
                images = []
                
                for img_path in glob.glob(image_path):                    
                    # current day
                    today = str(date.today())
                    #################
                    
                    # current datetime
                    time = (str(datetime.now().year), str(datetime.now().month), str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), str(datetime.now().second), str(datetime.now().microsecond))
                    current_time = ''.join(time)
                    #################
                    
                    name = current_time + str(image_path.split('/')[-1])
                    
                    img = cv.imread(img_path)
                    img1 = np.array(Image.open(img_path))[:, :, :3]
                    open_file = img1/255.0
                    
                    resize = cv.resize(open_file, (256, 256))
                    resize2 = cv.resize(img, (256, 256))
                    resize2 = cv.cvtColor(resize2, cv.COLOR_BGR2RGB)
                    resized = Image.fromarray(resize2)
                    resized_save = os.path.join('static/resized', name)
                    resized.save(resized_save)
                    
                    images.append(resize)
                    img = np.expand_dims(images[0], axis=0)
                    pred = make_pred_good(model(img))
                    pred[pred>0.5] = 1.0
                    pred[pred<0.5] = 0.0
                    
                    predicted = Image.fromarray((pred * 255).astype(np.uint8))
                    
                    number_of_white_pix = np.sum(pred == 1.0)
                    proper_area =  round(number_of_white_pix/(256*256*3)*100, 2)
                                    
                    #predicted = Image.fromarray(np.array(pred), 'GRAY')
                    predicted_save = os.path.join('static/mask', name)
                    predicted.save(predicted_save)
                        
                    print('previsao feita')
                    result = placeMaskOnImg(img[0], pred)
                    im = Image.fromarray((result * 255).astype(np.uint8))
                    result_save = os.path.join('static/predictions', name)    
                    im.save(result_save)
                    try:
                        dir = 'static/images'
                        for f in os.listdir(dir):
                            os.remove(os.path.join(dir, f))
                    except:
                        print("The system cannot find the file specified")
                        
                    conn = get_db_connection()
                    conn.execute('INSERT INTO image (image_name, description, image_path, mask_path, result_path, proper_area, date_upload, polygon, area, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                (name, description, resized_save, predicted_save, result_save, proper_area, today, polygon, area_m2, id))
                    conn.commit()
                    last_id = list(conn.execute('SELECT max(id) FROM image').fetchone())
                    conn.close()
                    return redirect(url_for('predict', id=last_id[0]))
                
            elif request.form['button'] == "Ver mais":
                id_ = request.form['id']
                return redirect(url_for('predict', id=id_))
            
            elif request.form['button'] == "Deletar":
                id_ = request.form['id']
                conn = get_db_connection()
                conn.execute('DELETE FROM image where id = (?)', [id_])
                conn.commit()
                posts = conn.execute('SELECT * FROM image where user_id = (?)', [id]).fetchall()
                conn.close()
                return render_template('index.html', posts=posts)
                
            elif request.form['button'] == "Pesquisar":
                date_start = request.form['date_start']
                date_end = request.form['date_end']
                
                select = request.form.get('comp_select')
                if select == '1':
                    query = 'SELECT * FROM image where user_id = (?) and proper_area <= 25'
                elif select == '2':
                    query = 'SELECT * FROM image where user_id = (?) and proper_area > 25 and proper_area <= 50'
                elif select == '3':
                    query = 'SELECT * FROM image where user_id = (?) and proper_area > 50 and proper_area <= 75'
                elif select == '4':
                    query = 'SELECT * FROM image where user_id = (?) and proper_area > 75 and proper_area <= 100'
                else:
                    query = 'SELECT * FROM image where user_id = (?)'
                
                conn = get_db_connection()
                
                if date_start != '' and date_end != "":
                    print(query + " and date_upload >= '"+str(date_start)+"' and date_upload <= '"+str(date_end)+"'")
                    posts = conn.execute(query + " and date_upload >= '"+str(date_start)+"' and date_upload <= '"+str(date_end)+"'", [id]).fetchall()
                else:
                    posts = conn.execute(query, [id]).fetchall()
                print(list(posts))
                conn.close()
                return render_template('index.html', posts=posts)
                
            return redirect(url_for('predict', id=0))
    else:
        return redirect(url_for('logout'))
        

@app.route("/predict/<id>", methods=('GET', 'POST'))
def predict(id):

    if request.method == 'GET':
        itens = []
        conn = get_db_connection()
        images = conn.execute('SELECT * FROM image where id = (?)', [id]).fetchone()
        
        if len(images['polygon']) > 0:
            polygon = images['polygon']
            polygon = polygon[2:-2]
            polygon = list(eval(polygon))
            res = [list(ele) for ele in polygon]
        else:
            res = ""
                    
        dict_imgs = {
            "id" : images['id'],
            "image_name": images['image_name'],
            "description": images['description'], 
            "image_path": images['image_path'],
            "mask_path": images['mask_path'],
            "result_path": images['result_path'],
            "proper_area": images['proper_area'],
            "date_upload": images['date_upload'],
            "polygon": res,
            "area": images['area'] ,
            "user_id": images['user_id']
        }

        return render_template('predict.html', posts=dict_imgs)
    
    # elif request.method == 'POST':
    #     conn = get_db_connection()
    #     images = conn.execute('SELECT * FROM image where id = (?)', [id]).fetchone()
    #     print(list(images))
    #     conn.close()
    #     return render_template('predict.html', posts=images)

if __name__ == "__main__":
    app.debug = True
    app.run()