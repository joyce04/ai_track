import smtplib
from email.mime.text import MIMEText

def send_email():
    HOST = 'smtp.naver.com'
    _from = '보내는 네이버 주소'
    _to = '받는 주소'

    contents = """Hello, your server just died."""

    msg = MIMEText(contents, _charset='utf-8')
    msg['Subject'] = '[ALERT]'
    msg['From'] = _from
    msg['To'] = _to

    host = smtplib.SMTP_SSL(HOST, 465)
    host.login('네이버 아이디', '네이버 비번')
    host.sendmail(_from, [_to], msg.as_string())
    host.quit()

send_email()
