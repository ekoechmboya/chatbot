import secrets
import hashlib
from django.urls import reverse
from django.shortcuts import render, redirect
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_encode 
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.core.mail import EmailMessage
from validate_email import validate_email
from django.contrib.auth.models import User
from django.contrib import messages

def generate_token(user):
    token = secrets.token_urlsafe(40) 
    user.activation_token = hashlib.sha256(token.encode('utf-8')).hexdigest() 
    return token



def send_activation_email(user, request):
    current_site = get_current_site(request)
    email_subject = 'Activate your account'
    email_body = render_to_string('main/activate.html', {
        'user': user,
        'domain': current_site.domain,
        'uid': urlsafe_base64_encode(force_bytes(user.pk)),
        'token': generate_token(user)  
    })

    email = EmailMessage(
        subject=email_subject,
        body=email_body,
        from_email='ekoech.mboya@gmail.com', 
        to=[user.email]
    )

    email.send()



    
def activate_user(request, uidb64, token):
    try:
       
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)

    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user and generate_token.check_token(user, token):
        
        user.is_active = True
        user.save()

        messages.success(request, 'Your account has been activated. You can now log in.')
        return redirect(reverse('login'))  

    else:
        
        messages.error(request, 'Invalid activation link.')
        return render(request, 'main.activate-failed.html') 
