from datetime import timezone
import os
import re
from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse
from HumanChatbot import settings
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from .aifunctions import answer_from_pdf_embeddings, fetch_and_store_embeddings, generate_feedback_response, get_relevant_context, process_pdf_embeddings, format_ai_response, ask_ai_with_context
from django.template.loader import render_to_string
from django.utils import timezone 
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from .models import (
    Conversation, Message, AIResponse, UserFeedback, 
    Intent, Entity, ChatSession, InteractionLog
)
from django.core.files.storage import default_storage
from .forms import PDFUploadForm


# Initialize ChatOpenAI model
model = ChatOpenAI(model="gpt-4o", api_key="")

# Initialize chat history with an optional system message
humanize_prompt = """Please respond as a friendly human assistant. When you reply, always begin with a greeting using your name (you can choose a friendly, human name like 'Alex' or 'Jamie'). You do not have to mention your
         name every time if the conversation is continuing. Avoid mentioning that you are an AI or language model; instead, act as though you are 
          a knowledgeable human helper. Use natural, empathetic language, and personalize your response to make the person feel comfortable.
          Give pure text responses without styling like bolding and adding new lines.
            If they ask about your background, keep it simple and human-like. Always be clear, positive, and relatable, offering practical advice or information in a warm and approachable tone."""
chat_history = []
chat_history.append(HumanMessage(humanize_prompt))

def chat_page(request):
    user = request.user
    print(user)

    return render(request, "chatbot/chat_page.html")


@csrf_exempt
def send_message(request):
    if request.method == "POST" and request.FILES:
        # Check if the PDF file exists in the uploaded files
        if "pdf_file" not in request.FILES:
            return JsonResponse({"status": "error", "message": "No file uploaded. Please upload a PDF file."})

        pdf_file = request.FILES["pdf_file"]

        # Validate if the uploaded file is a PDF
        if not pdf_file.name.endswith(".pdf"):
            return JsonResponse({"status": "error", "message": "Invalid file type. Please upload a PDF file."})

        # Step 1: Ensure the upload directory exists
        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        # Step 2: Save the uploaded file using Django's default storage
        file_path = os.path.join(upload_dir, pdf_file.name)
        
        # Check if the file already exists and handle naming conflict
        if default_storage.exists(file_path):
            base, ext = os.path.splitext(pdf_file.name)
            counter = 1
            while default_storage.exists(file_path):
                file_path = os.path.join(upload_dir, f"{base}_{counter}{ext}")
                counter += 1

        # Step 3: Save the file
        with default_storage.open(file_path, "wb+") as destination:
            for chunk in pdf_file.chunks():
                destination.write(chunk)

        # Step 4: Respond with a success message
        return JsonResponse({
            "status": "success",
            "message": "PDF file uploaded successfully.",
            "file_path": file_path  # Include file path in the response if needed
        })

    if request.method == "POST":
        chat_type = request.POST.get("chat_type")
        user_message = request.POST.get("message")
        selected_chat_type = "text"

        # Step 1: Assign chat type if `chat_type` is provided in the request
        if chat_type:
            # End any existing active conversation for the user
            Conversation.objects.filter(user=request.user, ended_at=None).update(ended_at=timezone.now())

            # Create a new conversation for the selected chat type
            new_conversation = Conversation.objects.create(user=request.user)

            # Store the selected chat type and conversation ID in the session
            request.session['selected_chat_type'] = chat_type
            request.session['conversation_id'] = new_conversation.id

            return JsonResponse({"status": "success", "message": f"Chat type set to {chat_type}"})

        # Step 2: Process message based on the selected chat type
        selected_chat_type = request.session.get('selected_chat_type')
        conversation_id = request.session.get('conversation_id')

        if selected_chat_type:
            conversation = Conversation.objects.filter(id=conversation_id).first()
            messages_html = ""

            if selected_chat_type == "text":
                if user_message:
                    user_message_content = user_message

                    # Create and save the user's Message
                    user_message = Message.objects.create(
                        conversation=conversation,
                        sender="user",
                        text=user_message_content
                    )

                    # Append user message to chat 
                    chat_history.append(HumanMessage(user_message_content))

                    # Generate AI response
                    result = model.invoke(chat_history)
                    ai_response_content = result.content
                    ai_response_content = format_ai_response(ai_response_content)

                    chat_history.append(SystemMessage(ai_response_content))

                    # Create and save the AI Message
                    ai_message = Message.objects.create(
                        conversation=conversation,
                        sender="bot",
                        text=ai_response_content
                    )

                    # Save the AI response details in AIResponse model
                    AIResponse.objects.create(
                        message=ai_message,
                        response_text=ai_response_content,
                    )

                    # Log the interaction
                    InteractionLog.objects.create(
                        user=request.user,
                        event_type="message_sent",
                        data={"content": user_message_content}
                    )
                    InteractionLog.objects.create(
                        user=request.user,
                        event_type="message_received",
                        data={"content": ai_response_content}
                    )

                    # Retrieve only new messages created after the last message ID saved in session
                    last_message_id = request.session.get("last_message_id", 0)
                    new_messages = Message.objects.filter(
                        conversation=conversation,
                        id__gt=last_message_id
                    ).order_by("created_at")

                    # Render new messages to HTML in correct order
                    messages_html = ''.join([
                        render_to_string("chatbot/message.html", {"message": message.text, "sender": message.sender})
                        for message in new_messages
                    ])

                    # Update session with the latest message ID
                    if new_messages.exists():
                        request.session["last_message_id"] = new_messages.last().id

                    return HttpResponse(messages_html)

            elif selected_chat_type == "feedback":
                user_feedback_content = user_message
                feedback_response = generate_feedback_response(user_feedback_content)

                # Save feedback in UserFeedback model
                UserFeedback.objects.create(
                    conversation=conversation,
                    comments=user_feedback_content,
                    created_at=timezone.now()
                )
                # Create and save the user's Message
                user_message = Message.objects.create(
                    conversation=conversation,
                    sender="user",
                    text=user_feedback_content
                )

                # Log the feedback submission
                InteractionLog.objects.create(
                    user=request.user,
                    event_type="feedback_submitted",
                    data={"content": user_feedback_content}
                )
                # Create and save the AI Message
                ai_message = Message.objects.create(
                    conversation=conversation,
                    sender="bot",
                    text=feedback_response
                )

                # Save the AI response details in AIResponse model
                AIResponse.objects.create(
                    message=ai_message,
                    response_text=feedback_response,
                )

                # Retrieve only new messages created after the last message ID saved in session
                last_message_id = request.session.get("last_message_id", 0)
                new_messages = Message.objects.filter(
                    conversation=conversation,
                    id__gt=last_message_id
                ).order_by("created_at")

                # Render new messages to HTML in correct order
                messages_html = ''.join([
                    render_to_string("chatbot/message.html", {"message": message.text, "sender": message.sender})
                    for message in new_messages
                ])

                # Update session with the latest message ID
                if new_messages.exists():
                    request.session["last_message_id"] = new_messages.last().id

                return HttpResponse(messages_html)


            elif selected_chat_type == "pdf":
                ai_response_content = answer_from_pdf_embeddings(humanize_prompt + user_message)

                if not ai_response_content:
                    ai_response_content = "I'm sorry, but I couldn't generate a response based on the provided PDF content."

                user_message = Message.objects.create(
                    conversation=conversation,
                    sender="user",
                    text=user_message
                )

                ai_message = Message.objects.create(
                    conversation=conversation,
                    sender="bot",
                    text=ai_response_content
                )

                AIResponse.objects.create(
                    message=ai_message,
                    response_text=ai_response_content,
                )

                InteractionLog.objects.create(
                    user=request.user,
                    event_type="pdf_question",
                    data={"source": "pdf", "question": user_message.text}
                )

                # Retrieve only new messages created after the last message ID saved in session
                last_message_id = request.session.get("last_message_id", 0)
                new_messages = Message.objects.filter(
                    conversation=conversation,
                    id__gt=last_message_id
                ).order_by("created_at")

                # Render new messages to HTML in correct order
                messages_html = ''.join([
                    render_to_string("chatbot/message.html", {"message": message.text, "sender": message.sender})
                    for message in new_messages
                ])

                # Update session with the latest message ID
                if new_messages.exists():
                    request.session["last_message_id"] = new_messages.last().id

                return HttpResponse(messages_html)

            elif selected_chat_type == "website":
                url_pattern = r'https?://\S+'
                urls = re.findall(url_pattern, user_message)
                if urls:
                    fetch_and_store_embeddings(user_message)

                context = get_relevant_context(user_message)
                ai_response = ask_ai_with_context(context,humanize_prompt + user_message)
                ai_response_content = ai_response.content

                if not ai_response_content:
                    ai_response_content = "I'm sorry, but I couldn't generate a response based on the provided content."

                user_message = Message.objects.create(
                    conversation=conversation,
                    sender="user",
                    text=user_message
                )

                ai_message = Message.objects.create(
                    conversation=conversation,
                    sender="bot",
                    text=ai_response_content
                )

                AIResponse.objects.create(
                    message=ai_message,
                    response_text=ai_response_content,
                )

                InteractionLog.objects.create(
                    user=request.user,
                    event_type="website_question",
                    data={"question": user_message.text}
                )
                # Retrieve only new messages created after the last message ID saved in session
                last_message_id = request.session.get("last_message_id", 0)
                new_messages = Message.objects.filter(
                    conversation=conversation,
                    id__gt=last_message_id
                ).order_by("created_at")

                # Render new messages to HTML in correct order
                messages_html = ''.join([
                    render_to_string("chatbot/message.html", {"message": message.text, "sender": message.sender})
                    for message in new_messages
                ])

                # Update session with the latest message ID
                if new_messages.exists():
                    request.session["last_message_id"] = new_messages.last().id

                return HttpResponse(messages_html)

        # If no chat type is selected, return an HTTP response with an error message
        error_message_html = render_to_string("chatbot/message.html", {
            "message": "Please select a chat type to continue.",  # Error message content
            "sender": "bot"  # Sender set to "bot"
        })

        return HttpResponse(error_message_html)


def handle_pdf_upload(request):
    if request.method == "POST" and request.FILES:
        form = PDFUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            # Save the file using the model form
            uploaded_pdf = form.save()
            
            # Construct the absolute URL of the file
            source_name = uploaded_pdf.pdf_file.name  # Use filename as identifier
            # Get the absolute file path on the server
            upload_path = uploaded_pdf.pdf_file.path
            print(upload_path, source_name)
            process_pdf_embeddings(upload_path, source_name)
            # Return a success message we absolute URL
            return redirect("chat_page")
        else:
            return JsonResponse({"status": "error", "message": "Invalid file. Please upload a valid PDF."})
    
    # If GET request or invalid form, render the form
    form = PDFUploadForm()
    return render(request, 'chatbot/upload_pdf.html', {'form': form})
