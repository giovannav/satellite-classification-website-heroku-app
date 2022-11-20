import io
import dominate
from dominate.tags import *
import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from itsdangerous import URLSafeTimedSerializer

sender = 'giovannafrederico60@gmail.com'


def generate_page(token):

    doc = dominate.document(title='My web page')

    with doc:
        with header():
            h1('Solicitação de redefinicação de senha', cls='title-class')
            
        with div(cls='song-class'):
            p(b('Token: '), token)

    return doc


def send_mail(receiver, token):
    recipients = list(receiver.split(','))
    my_page = str(generate_page(token))
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ',' .join(recipients)
    msg['Subject'] = 'Recuperar minha senha'
    msg.attach(MIMEText(my_page, 'html'))
    msg = msg.as_string()
    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(sender, 'qcbsbcqwgmpmskgx')
        server.sendmail(sender, recipients, msg)
        server.quit()
        print('Sent')
    except:
        print('Failed')
        