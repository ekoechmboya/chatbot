from django.shortcuts import render, redirect
from django.contrib import messages
from validate_email import validate_email
from django.contrib.auth.models import User
from .utils import generate_token, send_activation_email, activate_user
from django.contrib.auth import authenticate, login, logout



def login_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome, {user.username}!')
            return redirect('chat')
        else:
            messages.error(request, 'Invalid username or password')

    return render(request, 'authentication/login.html')


def register(request):
    if request.method == "POST":
        context = {'has_error': False, 'data': request.POST}
        email = request.POST.get('email')
        username = request.POST.get('username')
        password = request.POST.get('password')
        password2 = request.POST.get('password2')

        if len(password) < 6:
            messages.add_message(request, messages.ERROR, 'Password should be at least 6 characters')
            context['has_error'] = True

        if User.objects.filter(username=username).exists():
            messages.add_message(request, messages.ERROR, 'Username is taken, choose another one')
            context['has_error'] = True

        if User.objects.filter(email=email).exists():
            messages.add_message(request, messages.ERROR, 'Email is taken, choose another one')
            context['has_error'] = True

        if context['has_error']:
            return render(request, 'authentication/register.html', context)

        user = User.objects.create_user(username=username, email=email)
        user.set_password(password)
        user.save()
        return redirect('login')

    return render(request, 'authentication/register.html')


def logout_view(request):
    logout(request)
    return redirect('login')

def chat_home(request):
    return render(request, 'chatbot/chat.html')

