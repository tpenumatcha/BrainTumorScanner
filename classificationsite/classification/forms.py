from django import forms


class Records(forms.Form):
    name = forms.CharField(label="Name", max_length=200)
    Date_of_Birth = forms.DateField(required=False)
    scan = forms.ImageField()