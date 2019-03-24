import string
import re
import dropbox
import sys
from pickle import dump
from unicodedata import normalize
from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().replace('!', '.').replace('?', '.').replace(',', '.').split('.')
    pairs = [line.split('\t') for line in lines]
    return pairs


# clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved txt as pkl: %s' % filename)


# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


LOCALFILE = 'translated_file.txt'
BACKUPPATH = '/dropboxfile_translated.txt'


# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
    predicted = ''
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_target = raw_dataset[i, 0]
        if i < 10:
            print('     target=[%s], predicted=[%s]' % (raw_target, translation))
        predicted = predicted + translation + '. '
    # print(predicted)
    print('Writing translated file to [%s]' % LOCALFILE)
    f = open(LOCALFILE, "w+")
    f.write(predicted)


# Uploads contents of LOCALFILE to Dropbox
def upload(LOCALFILE):
    with open(LOCALFILE, 'rb') as f:
        # We use WriteMode=overwrite to make sure that the settings in the file
        # are changed on upload
        print("Uploading " + LOCALFILE + " to Dropbox as " + BACKUPPATH + "...")
        try:
            dbx.files_upload(f.read(), BACKUPPATH, mode=WriteMode('overwrite'))
        except ApiError as err:
            # This checks for the specific error where a user doesn't have
            # enough Dropbox space quota to upload this file
            if (err.error.is_path() and
                    err.error.get_path().reason.is_insufficient_space()):
                sys.exit("ERROR: Cannot back up; insufficient space.")
            elif err.user_message_text:
                print(err.user_message_text)
                sys.exit()
            else:
                print(err)
                sys.exit()


filename_db_download = '/dropboxfile.txt'
filename_db_local = 'local_dropboxfile.txt'
### ========== TODO : START DOWNLOAD TXT FILE FROM DROPBOX ========== ###
TOKEN = 'Igo5S04-xwAAAAAAAAAACErc46g4UgAYqazxan531DGy6xEiLSLR_xt-yDrzuAvy'
dbx = dropbox.Dropbox(TOKEN)
try:
    print(dbx.users_get_current_account())
except AuthError:
    sys.exit("ERROR: Invalid access token; try re-generating an ""access token from the app console on the web.")
print('Downloading [%s] from Dropbox...' % filename_db_download)
dbx.files_download_to_file(filename_db_local, filename_db_download)
print(
    'Successfully downloaded [%s] from Dropbox and saved locally as [%s].' % (filename_db_download, filename_db_local))
### ========== TODO : END DOWNLOAD TXT FILE FROM DROPBOX ========== ###


### ========== TODO : START CLEAN TXT FILE FROM DROPBOX ========== ###
print('Cleaning [%s]...' % filename_db_local)
# load dataset
filename = filename_db_local
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, 'dropboxfile.pkl')
# spot check
print(clean_pairs[:, :])
print('Successfully cleaned [%s].' % filename_db_local)
### ========== TODO : END CLEAN TXT FILE FROM DROPBOX ========== ###

### ========== TODO : START TRANSLATION ========== ###
print('Translating [%s] file...' % filename_db_local)
# load datasets
dataset = load_clean_sentences('english-spanish-both.pkl')
train = load_clean_sentences('dropboxfile.pkl')
# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# prepare spanish tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
# prepare data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 0])
# load model
model = load_model('model.h5')
# test on some training sequences
evaluate_model(model, eng_tokenizer, trainX, train)
print('Successfully translated [%s] file.' % filename_db_local)
### ========== TODO : END TRANSLATION ========== ###

### ========== TODO : START UPLOAD TRANSLATED FILE ========== ###
print('Uploading [%s] file to Dropbox...' % LOCALFILE)
upload(LOCALFILE)
print('Successfully uploaded [%s] file to Dropbox as [%s]' % (LOCALFILE, BACKUPPATH))
### ========== TODO : END UPLOAD TRANSLATED FILE ========== ###