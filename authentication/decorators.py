from django.shortcuts import redirect


def logout_required(view_func):
    def wrapper(request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect('home')  # Redirect authenticated user to home
        return view_func(request, *args, **kwargs)
    return wrapper

def superuser_required(view_func):
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_superuser:
            # Redirect unauthenticated users or non-superusers elsewhere
            return redirect('home')  # Change 'home' to the desired URL name or path
        return view_func(request, *args, **kwargs)
    return wrapper

def staffuser_required(view_func):
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_staff:
            # Redirect unauthenticated users or non-staff users elsewhere
            return redirect('home')  # Change 'home' to the desired URL name or path
        return view_func(request, *args, **kwargs)
    return wrapper