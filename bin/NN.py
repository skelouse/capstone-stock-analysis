def repeat_for_n_days():
    return 1

class day3():
    def static_company(self):
        return repeat_for_n_days()

    def predict_analyst(self):
        return repeat_for_n_days()

    def predict_performance(self):
        return repeat_for_n_days()

    def predict_price(self):
        return repeat_for_n_days()




class day2():
    def predict_price(self):
        return day3.predict_price()

    def predict_performance(self):
        return day3.predict_analyst(), day3.predict_performance()

    def predict_analyst(self):
        return day3.predict_price()

    def static_company(self):
        return day3.predict_analyst(), day3.static_company()


class day1():
    def predict_price(self):
        return day2.predict_price()

    def predict_performance(self):
        return day2.predict_analyst(), day2.predict_performance()

    def predict_analyst(self):
        return day2.predict_price()

    def static_company(self):
        return day2.predict_analyst(), day2.static_company()

class Data():
    def price(self):
        return day1.predict_price()

    def performance(self):
        return day1.predict_analyst(), day1.predict_performance()

    def analyst(self):
        return day1.predict_price()

    def static_company(self):
        return day1.predict_analyst(), day1.static_company()


def send_data():
    Data.price()
    Data.performance()
    Data.analyst()
    Data.static_company()