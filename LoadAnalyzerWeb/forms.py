# -*- coding: utf-8 -*-
from django import forms


import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
	


class SelectTrackForm(forms.Form):
    docfile = forms.FileField(
        label='Select a file'
    )

class FileListForm(forms.Form):
	def __init__(self, *args, **kwargs):
		choices = kwargs.pop('choices')
		super(FileListForm, self).__init__(*args, **kwargs)
		self.fields["filelist"] = forms.ChoiceField(choices=choices, label='Track list')

