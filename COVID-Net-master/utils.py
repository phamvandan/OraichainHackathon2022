
data = open('./demo_cxr.txt')
text_lines = data.readlines()

def do_predict(filename):
    response = {}
    response['filename'] = filename
    response['serverity level'] = 1
    response['confidence'] = 0.9
    for text_line in text_lines:
        elements = text_line.split('\t')
        if elements[0] == filename:
            response['filename'] = elements[0]
            response['serverity level'] = elements[1]
            response['confidence'] = elements[2].replace('\n', '')
            break
    return response