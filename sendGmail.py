import yagmail
import Constant as  Constants
import datetime

def sendAlertEmail(person_name=Constants.person,  family_member_email_ids=[Constants.email]):
    try:
        # Get the current timestamp
        current_time = datetime.datetime.now()

        # Format the timestamp
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # Include the formatted timestamp in the message
        message = 'Hello, \nContact {} as soon as possible because {} is found in a drowsy situation while driving. This message was sent by the Drowsiness Detection System at {}.'.format(person_name, person_name, formatted_time)

        # Create a yagmail SMTP connection
        yag = yagmail.SMTP("autoemailsender2@gmail.com", "tczewxnxfrpviped")

        # Send the email to family members
        yag.send(to=family_member_email_ids, subject='Drowsiness Alert Message', contents=message)

        # Close the yagmail SMTP connection
        yag.close()

        print("Email sent successfully")
    except Exception as e:
        print('Email not sent due to:', e)
