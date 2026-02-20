from enum import Enum


class Insurer(Enum):
    SAMSUNG = ("samsung", "삼성화재")
    KB = ("kb", "KB손해보험")
    MERITZ = ("meritz", "메리츠화재")
    ETC = ("etc", "기타")

    def __init__(self, code, kr_name):
        self.code = code
        self.kr_name = kr_name

    @classmethod
    def from_code(cls, code):
        for member in cls:
            if member.code == code:
                return member
        return cls.ETC
