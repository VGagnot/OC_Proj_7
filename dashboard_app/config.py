import os

SECRET_KEY = '56uYe^$+|-y]:3[*,^s)@-Z8'

basedir = os.path.abspath(os.path.dirname(__file__))
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')