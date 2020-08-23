

class data:

    def __init__(self,temperature,heartrate,gender):
        self.temperature = temperature
        self.heartrate = heartrate
        self.gender = gender

    def asVecrtorP(self):
        vec=[self.temperature,self.heartrate]
        return vec

