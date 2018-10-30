from django.db import models

class Process(models.Model):
	query_text = models.CharField(max_length=200)

	def __str__(self):
		return self.query_text
