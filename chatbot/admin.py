from django.contrib import admin
from .models import (
    DocumentEmbedding,
    Conversation,
    Message,
    AIResponse,
    UserFeedback,
    Intent,
    UploadedPDF,
    Entity,
    ChatSession,
    InteractionLog,
)

# Register models one by one
admin.site.register(DocumentEmbedding)
admin.site.register(Conversation)
admin.site.register(Message)
admin.site.register(AIResponse)
admin.site.register(UserFeedback)
admin.site.register(Intent)
admin.site.register(UploadedPDF)
admin.site.register(Entity)
admin.site.register(ChatSession)
admin.site.register(InteractionLog)
