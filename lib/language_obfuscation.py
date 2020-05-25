import requests
from requests import post

class LanguageObfuscation(object):
    def __init__(self,languages):
        self.key = 'trnsl.1.1.20200525T135708Z.1ef84ed1d61c6d83.80e194b493bda75e7e8089435d5e3517ee97d6a9'
        self.url = "https://translate.yandex.net/api/v1.5/tr.json/translate"

        self.languages = languages
        
    def main_language_obfuscation(self,sentence):
        for language in self.languages:
            sentence_temp = self.send_response(language,sentence)
            if sentence_temp == sentence:
                return sentence
            else:
                sentence = sentence_temp
        return sentence_temp
    
    def send_response(self,lang,data_to_transfer):
        dataRequest = {'key': self.key,
                       'lang':lang,
                       'text':data_to_transfer}

        result = post(data=dataRequest, url=self.url)

        jsonResponseData = result.json()
        
        try:
            text = jsonResponseData['text'][0]
        except:
            return data_to_transfer
        return text