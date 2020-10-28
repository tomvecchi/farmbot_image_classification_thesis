"""
    Created by Tom Vecchi for 2020 UQ Farmbot Thesis Project

    Function for emailing the final output message of the program.
"""

import smtplib, ssl
import creds

# Sends an email to the specified address. Defaults to my email address- please change 
# this if needed.
# Note: this is an insecure way of authenticating. Do not use a personal email account
# as the sending address.
def send_email(message, receiver_email="macemperor2@gmail.com"):

    sender_email = creds.gmail_account
    password = creds.gmail_password 
    #more secure than just hardcoding it in a py file

    # Create a secure SSL context
    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            message = message + "\r\n Sent from my Python"
            server.sendmail(sender_email, receiver_email, message)
            print("Successfully sent email")
        
    except Exception as error:
        print(error)



