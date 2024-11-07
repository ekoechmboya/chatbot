from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_page, name='chat_page'),  # Main chat interface
    path('chat/upload/', views.handle_pdf_upload, name='handle_pdf_upload'),
    path('chat/send_message/', views.send_message, name='send_message'),

    # path('send-message/', views.send_message, name='send_message'),  # Endpoint for sending messages
    # path('get-response/', views.get_response, name='get_response'),  # Endpoint for real-time AI responses
    # path('end-conversation/', views.end_conversation, name='end_conversation'),  # Endpoint to end the conversation
    # path('submit-feedback/', views.submit_feedback, name='submit_feedback'),  # Endpoint for submitting feedback
]
