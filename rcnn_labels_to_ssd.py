import csv
with open('annotations/test_labels_yolo.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(["filename","width","height","class","xmin","ymin","xmax","ymax"])
    with open('annotations/test_labels.csv', 'r') as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:
            if i != 0:
                dw = 1./float(row[1]) # 1/width
                dh = 1./float(row[2]) # 1/height
                x = (float(row[4]) + float(row[6]))/2.0 # (xmin + xmax)/2
                y = (float(row[5]) + float(row[7]))/2.0 # (ymin + ymax)/2
                w = float(row[6]) - float(row[4]) # xmax - xmin
                h = float(row[7]) - float(row[5]) # ymax - ymin
                x = x*dw # (xmin + xmax)/2 * 1/width
                y = y*dh # (ymin + ymax)/2 * 1/height
                w = w*dw # (xmax - xmin) * 1/width
                h = h*dh # (ymax - ymin) * 1/height
                writer.writerow([row[0],row[1],row[2],row[3],str(x),str(y),str(w),str(h)])
            i+=1