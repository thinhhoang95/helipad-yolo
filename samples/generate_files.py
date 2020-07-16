with open('helipad.txt', 'a') as file:
    for i in range(1,230,1):
        filename = '{:03d}'.format(i)
        file.write('images/img_' + filename + '.jpg\n')
