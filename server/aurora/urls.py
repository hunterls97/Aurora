from django.conf.urls import url, include

from . import views

app_name = 'aurora'
urlpatterns = [
	url(r'^index/$', views.index, name='index'),
	url(r'^train/$', views.train, name='train'),
]