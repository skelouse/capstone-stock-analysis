import os
import json
import time
import requests
from dotenv import load_dotenv


load_dotenv()
db_name = os.getenv('FB_NAME')
web_API_key = os.getenv('WAK')
email = os.getenv('EMAIL')
password = os.getenv('PASSWORD')
signin_url = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/verifyPassword?key="  # noqa


class ConnectionError(Exception):
    pass


class FireBase():
    wak = web_API_key
    url = f"https://{db_name}.firebaseio.com/"
    headers = {"Content-type": "application/json"}

    def __init__(self):
        self.today = (time.strftime("%m-%d-%Y"))
        self.sign_in()

    def put_new_data(self, data):
        json_data = json.dumps(data)
        req = requests.update(
            self.url
            + "data/"
            + ".json?auth="
            + self.id_token,
            data=json_data,
            headers=self.headers
        )
        if req.ok:
            return ({'success': True, 'req': req})
        else:
            self.capture_bug(req.text)
            return ({'success': False, 'error': req})

    def refresh_all_data(self, data):
        json_data = json.dumps(data)
        req = requests.patch(
            self.url
            + "data/"
            + ".json?auth="
            + self.id_token,
            data=json_data,
            headers=self.headers
        )
        if req.ok:
            return ({'success': True, 'req': req})
        else:
            self.capture_bug(req.text)
            return ({'success': False, 'error': req})

    def refresh_index(self, data):
        json_data = json.dumps(data)
        req = requests.patch(
            self.url
            + "index/"
            + ".json?auth="
            + self.id_token,
            data=json_data,
            headers=self.headers
        )
        if req.ok:
            return ({'success': True, 'req': req})
        else:
            self.capture_bug(req.text)
            return ({'success': False, 'error': req})

    def sign_in(self):
        signin_data = {
            "email": email,
            "password": password,
            "returnSecureToken": True}
        req = requests.post(signin_url+self.wak, data=signin_data)

        if req.ok:
            self.local_id = req.json()['localId']
            self.id_token = req.json()['idToken']
            return {'success': True}
        else:
            raise ConnectionError("Invalid Email or Password")

    def get_index(self):
        req = requests.get(
            self.url
            + "index/"
            + '.json?auth='
            + self.id_token)
        if req.ok:
            data = json.loads(req.content.decode())

            return ({'success': True, 'data': data})
        else:
            self.capture_bug(req.text)
            return ({'success': False, 'error': req})

    def get_sym_col(self, sym, col):
        req = requests.get(
            self.url
            + f"data/{sym}/{col}"
            + '.json?auth='
            + self.id_token)
        if req.ok:
            data = json.loads(req.content.decode())

            return ({'success': True, 'data': data})
        else:
            self.capture_bug(req.text)
            return ({'success': False, 'error': req})

    def capture_bug(self, traceback):
        undef = False
        quantity = 1
        logs_url = (self.url + 'logs/.json?auth=' + self.id_token)
        raw_exception = traceback.split('\n')
        main_error = raw_exception[-2]
        if len(main_error) >= 100:
            main_error = 'undef'
            undef = True
        logs_req = requests.get(logs_url)
        logs_data = json.loads(logs_req.content.decode())
        try:
            for i in logs_data.items():
                if i[0] == main_error:
                    quantity = (i[1]['quantity'] + 1)
        except AttributeError:
            pass
        if undef:
            data = json.dumps({
                main_error: {
                    'traceback': traceback,
                    'quantity': quantity,
                    'date': self.today
                    }})
            req = requests.post(logs_url, data=data)
        else:
            data = json.dumps({
                main_error: {
                    'traceback': traceback,
                    'quantity': quantity,
                    'date': self.today
                    }})

        print('exception sending')
        req = requests.patch(logs_url, data=data)
        print('exception sent = ', req.ok)


if __name__ == "__main__":
    fb = FireBase()
    req = fb.refresh_all_data({'test': 1})
