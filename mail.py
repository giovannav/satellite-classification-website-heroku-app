import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import dominate
from dominate.tags import *
import json

sender = 'giovannafrederico60@gmail.com'

def generate_page(token):

    doc = dominate.document(title='My web page')

    with doc:
        with header():
            h2('SIPH: Solicitação de redefinicação de senha', cls='title-class')
            
        with div(cls='song-class'):
            p('Olá! Uma requisição de definição de senha foi realizada no sistema SIPH.')
            p('Utilize o token abaixo para redefinir a senha:')
            p(b('Token: '), token)

    return doc


def send_mail(receiver, token):
    
    with open('static/files/credentials.json', 'r') as fcc_file:
     credentials = json.load(fcc_file)
    
    recipients = list(receiver.split(','))
    my_page = str(generate_page(token))
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ',' .join(recipients)
    msg['Subject'] = 'Recuperar minha senha: Sitema SIPH'
    msg.attach(MIMEText(my_page, 'html'))
    msg = msg.as_string()
    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(sender, credentials['token'])
        server.sendmail(sender, recipients, msg)
        server.quit()
        print('Sent')
    except:
        print('Failed')