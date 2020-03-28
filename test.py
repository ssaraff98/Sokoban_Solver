player = (1, 2)
box1 = (1, 3)
box2 = (3, 1)
boxes = [box1, box2]
a = tuple([player] + sorted(boxes))
print(a)