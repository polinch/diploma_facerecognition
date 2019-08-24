import testrecognition as fr

test = fr.CalculatePCA(5, 'train')
test.calc_pca()
print('PCA - 5 photos')
count_true = 0
for i in range(5):
    print('True answer: train' + str(i) + '.pgm')
    print('Test image: test' + str(2 * i) + '0.pgm')
    r = test.transform_faces('test' + str(2 * i))
    if r == i:
        count_true += 1
    print('Test image: test' + str(2 * i + 1) + '0.pgm')
    r = test.transform_faces('test' + str(2 * i + 1))
    if r == i:
        count_true += 1
    print('\n')
print('Count true:' + str(count_true) + ' / Count tests:' + str(2 * 5))
print('Quality = ' + str(count_true / 10 * 100) + '%')

test.pca_without_train(10, 'train')
print('PCA without new train - 10 photos')
for i in range(10):
    print('True answer: train' + str(i) + '.pgm')
    print('Test image: test' + str(2 * i) + '0.pgm')
    test.transform_faces('test' + str(2 * i))
    print('Test image: test' + str(2 * i + 1) + '0.pgm')
    test.transform_faces('test' + str(2 * i + 1))
    print('\n')