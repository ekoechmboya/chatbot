from datetime import timezone
from django.contrib.auth.models import User
from django.db import models
from django.db import models
from django.conf import settings
from django.utils import timezone 


class DocumentEmbedding(models.Model):
    content = models.TextField()  # Stores document content
    embedding = models.BinaryField()  # Stores the embedding vector as binary data
    source_url = models.URLField(null=True, blank=True)  # Optional field for the source URL
    metadata = models.JSONField(null=True, blank=True)  # Optional metadata for document

    def __str__(self):
        return f"Document from {self.source_url or 'unknown source'}"

class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"Conversation with {self.user.username} started at {self.started_at}"

    def end_conversation(self):
        self.ended_at = timezone.now()
        self.save()

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, related_name='messages', on_delete=models.CASCADE)
    sender = models.CharField(max_length=50)  
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender}: {self.text[:50]}..."

class AIResponse(models.Model):
    message = models.OneToOneField(Message, on_delete=models.CASCADE, related_name='ai_response')
    response_text = models.TextField()

class UserFeedback(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    comments = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class Intent(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(null=True, blank=True)
    
    def __str__(self):
        return self.name

class UploadedPDF(models.Model):
    # A field to store the file itself
    pdf_file = models.FileField(upload_to='uploads/pdfs/', verbose_name="PDF File")
    uploaded_at = models.DateTimeField(auto_now_add=True, verbose_name="Upload Time")
    
    def __str__(self):
        return self.pdf_file.name

class Entity(models.Model):
    name = models.CharField(max_length=100)
    entity_type = models.CharField(max_length=100)
    value = models.CharField(max_length=255)
    message = models.ForeignKey(Message, related_name='entities', on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.entity_type}: {self.value}"

class ChatSession(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=255, unique=True)
    active = models.BooleanField(default=True)
    last_interaction = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Session {self.session_id} for {self.user.username}"

class InteractionLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    event_type = models.CharField(max_length=50)  # e.g., 'message_sent', 'message_received'
    data = models.JSONField()  # Store additional event data as JSON
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.event_type} for {self.user.username} at {self.timestamp}"
