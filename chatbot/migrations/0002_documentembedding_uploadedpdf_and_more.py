# Generated by Django 5.1.2 on 2024-11-20 21:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chatbot", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="DocumentEmbedding",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("content", models.TextField()),
                ("embedding", models.BinaryField()),
                ("source_url", models.URLField(blank=True, null=True)),
                ("metadata", models.JSONField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="UploadedPDF",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "pdf_file",
                    models.FileField(
                        upload_to="uploads/pdfs/", verbose_name="PDF File"
                    ),
                ),
                (
                    "uploaded_at",
                    models.DateTimeField(auto_now_add=True, verbose_name="Upload Time"),
                ),
            ],
        ),
        migrations.RemoveField(
            model_name="airesponse",
            name="confidence_score",
        ),
        migrations.RemoveField(
            model_name="airesponse",
            name="response_time",
        ),
        migrations.RemoveField(
            model_name="userfeedback",
            name="rating",
        ),
    ]
