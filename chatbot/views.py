from datetime import timezone
import os
import re
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from HumanChatbot import settings
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from .aifunctions import answer_from_pdf_embeddings, fetch_and_store_embeddings, generate_feedback_response, get_relevant_context, process_pdf_embeddings, format_ai_response
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
chat_history = [SystemMessage(content="""Please respond as a friendly human assistant. When you reply, always begin with a greeting using your name (you can choose a friendly, human name like 'Alex' or 'Jamie').
          Avoid mentioning that you are an AI or language model; instead, act as though you are a knowledgeable human helper. Use natural, empathetic language, and personalize your response to make the person feel comfortable.
            If they ask about your background, keep it simple and human-like. Always be clear, positive, and relatable, offering practical advice or information in a warm and approachable tone.""")]

def chat_page(request):
    user = request.user
    print(user)

    # Step 1: Retrieve all conversations for the logged-in user
    conversations = Conversation.objects.filter(user=user).order_by("-created_at")  # Most recent first
    if conversations: 
        print("Conversations")
    else:
        print("no conversations")

    # Step 2: Generate a list of conversation titles
    conversation_list = []
    for conversation in conversations:
        first_message = conversation.messages.filter(sender="user").order_by("created_at").first()
        title = first_message.text[:100] if first_message else "Untitled Conversation"
        conversation_list.append({"id": conversation.id, "title": title})

    # Step 3: Handle the selected conversation (default to the first conversation if not selected)
    selected_conversation_id = request.GET.get("conversation_id")
    selected_conversation = None
    chat_history = []

    if selected_conversation_id:
        selected_conversation = Conversation.objects.filter(id=selected_conversation_id, user=user).first()
    elif conversations.exists():
        selected_conversation = conversations.first()  # Default to the latest conversation

    if selected_conversation:
        # Build chat history for the selected conversation
        for msg in selected_conversation.messages.all().order_by("created_at"):
            chat_history.append({"sender": msg.sender, "text": msg.text, "timestamp": msg.created_at})

    # Step 4: Prepare context for the template
    context = {
        "chat_history": chat_history,
        "conversations": conversation_list,
        "selected_conversation": selected_conversation.id if selected_conversation else None,
    }

    return render(request, "chatbot/chat_page.html", context)


@csrf_exempt
def send_message(request):
    # Check if the request method is POST and if a file is uploaded
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
                # Rename the file to avoid overwriting existing files
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
    # Check if this is a POST request to handle both chat type selection and message submission
    if request.method == "POST":
        chat_type = request.POST.get("chat_type")
        user_message = request.POST.get("message")
        message_text = user_message

        # Step 1: Assign chat type if `chat_type` is provided in the request
        if chat_type:
            request.session['selected_chat_type'] = chat_type  # Store chat type in session
            return JsonResponse({"status": "success", "message": f"Chat type set to {chat_type}"})

        # Step 2: Process message based on the selected chat type
        selected_chat_type = "text"
        selected_chat_type = request.session.get('selected_chat_type')
        
        if selected_chat_type:
            messages_html = {}
            if selected_chat_type == "text":
                if message_text:
                        user = request.user
                        user_message_content = request.POST.get("message")

                        # Retrieve or create a Conversation instance
                        conversation, created = Conversation.objects.get_or_create(
                            user=user,
                            ended_at=None  # Retrieve the active conversation
                        )

                        # Create and save the user's Message
                        user_message = Message.objects.create(
                            conversation=conversation,
                            sender="user",
                            text=user_message_content
                        )

                        # Append user message to chat history
                        chat_history = [HumanMessage(content=user_message_content)]

                        # Generate AI response
                        result = model.invoke(chat_history)
                        ai_response_content = result.content

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

                        # Update or create a ChatSession for the user
                        chat_session, _ = ChatSession.objects.update_or_create(
                            user=user,
                            active=True,
                            defaults={"last_interaction": timezone.now()}
                        )

                        # Log the interaction
                        InteractionLog.objects.create(
                            user=user,
                            event_type="message_sent",
                            data={"content": user_message_content}
                        )
                        InteractionLog.objects.create(
                            user=user,
                            event_type="message_received",
                            data={"content": ai_response_content}
                        )

                        # Retrieve only new messages created after the last message ID saved in session
                        last_message_id = request.session.get("last_message_id", 0)
                        new_messages = Message.objects.filter(
                            conversation=conversation,
                            id__gt=last_message_id
                        ).order_by("created_at")  # Ascending order, oldest to newes

                        # Render new messages to HTML in correct order
                        messages_html = ''.join([
                            render_to_string("chatbot/message.html", {"message": message.text, "sender": message.sender})
                            for message in new_messages
                        ])

                        # Update session with the latest message ID
                        if new_messages.exists():
                            request.session["last_message_id"] = new_messages.last().id

                        

                        # Send the ordered HTML response
                        return HttpResponse(messages_html)

            elif selected_chat_type == "feedback":
                # Feedback handling logic
                user = request.user
                user_feedback_content = message_text

                feedback_response = generate_feedback_response(user_feedback_content)
                ai_response_content = feedback_response
                # Retrieve the active conversation
                conversation = Conversation.objects.filter(user=user, ended_at=None).first()
                if not conversation:
                    # If no active conversation, create a new one for the feedback
                    conversation = Conversation.objects.create(user=user)
                
                # Save the feedback in the UserFeedback model
                UserFeedback.objects.create(
                    conversation=conversation,
                    comments=user_feedback_content,
                    created_at=timezone.now()
                )

                # Log the interaction for feedback submission
                InteractionLog.objects.create(
                    user=user,
                    event_type="feedback_submitted",
                    data={"content": user_feedback_content}
                )

                # Render confirmation message to HTML to display to user
                feedback_confirmation_html = render_to_string("chatbot/message.html", {
                    "message": ai_response_content, "sender:": "bot"})
                return HttpResponse(feedback_confirmation_html)

                pass
            elif selected_chat_type == "pdf":
                user = request.user
                message_content = message_text
                
                # Generate the AI response based on PDF embeddings
                ai_response_content = answer_from_pdf_embeddings(message_content)
                
                if not ai_response_content:  # Check if response is empty
                    ai_response_content = "I'm sorry, but I couldn't generate a response based on the provided PDF content."

                # Ensure message_content and ai_response_content are not empty
                if message_content.strip():
                    # Step 1: Log the user's question as a Message
                    conversation, _ = Conversation.objects.get_or_create(
                        user=user,
                        ended_at=None
                    )
                    user_message = Message.objects.create(
                        conversation=conversation,
                        sender="user",
                        text=message_content
                    )

                    # Step 2: Log the AI's response as a Message
                    ai_message = Message.objects.create(
                        conversation=conversation,
                        sender="bot",
                        text=ai_response_content
                    )

                    # Step 3: Save the response details in AIResponse
                    AIResponse.objects.create(
                        message=ai_message,
                        response_text=ai_response_content,
                    )

                    # Step 4: Log interactions in InteractionLog
                    InteractionLog.objects.create(
                        user=user,
                        event_type="pdf_question",
                        data={"source": "pdf", "question": message_content}
                    )

                    # Step 5: Render and return the response
                    response_html = render_to_string("chatbot/message.html", {
                        "message": ai_response_content, "sender": "bot"
                    })
                    
                    return HttpResponse(response_html)
                pass
            elif selected_chat_type == "website":
                user = request.user
                url_pattern = r'https?://\S+'
                urls = re.findall(url_pattern, message_text)
                
                if not urls:
                    return JsonResponse({"status": "error", "message": "No URL found in message."})
                pass

                url = urls[0]

                # Fetch and store embeddings from the URL content
                fetch_and_store_embeddings(message_text)

                # Retrieve relevant context from embeddings
                ai_response_content = get_relevant_context(message_text)
                
                if not ai_response_content:  # Check if response is empty
                    ai_response_content = "I'm sorry, but I couldn't generate a response based on the provided content."

                # Ensure message_text and ai_response_content are not empty
                if message_text.strip():
                    # Log the user's question as a Message
                    conversation, _ = Conversation.objects.get_or_create(
                        user=user,
                        ended_at=None
                    )
                    user_message = Message.objects.create(
                        conversation=conversation,
                        sender="user",
                        text=message_text
                    )
                    
                    # Log the AI's response as a Message
                    ai_message = Message.objects.create(
                        conversation=conversation,
                        sender="bot",
                        text=ai_response_content
                    )

                    # Save the response details
                    AIResponse.objects.create(
                        message=ai_message,
                        response_text=ai_response_content,
                    )

                    # Log interactions
                    InteractionLog.objects.create(
                        user=user,
                        event_type="website_question",
                        data={"url": url, "question": message_text}
                    )
                    
                    # Render AI response to HTML
                    website_response_html = render_to_string("chatbot/message.html", {
                        "message": ai_response_content, "sender": "bot"
                    })
                    
                    return HttpResponse(website_response_html)
                else:
                    return JsonResponse({"status": "error", "message": "Empty message text. Please provide content."})
                pass
            else:
                return JsonResponse({"status": "error", "message": "Invalid chat type selected."})
            

            return HttpResponse(messages_html)

        else:
            return JsonResponse({"status": "error", "message": "No chat type selected."})


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
            return JsonResponse({
                "status": "success",
                "message": "File uploaded successfully!",
            })
        else:
            return JsonResponse({"status": "error", "message": "Invalid file. Please upload a valid PDF."})
    
    # If GET request or invalid form, render the form
    form = PDFUploadForm()
    return render(request, 'chatbot/upload_pdf.html', {'form': form})
