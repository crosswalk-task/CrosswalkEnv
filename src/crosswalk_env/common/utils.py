def is_overlapping(left1, right1, left2, right2):
    if right1 < left2 or right2 < left1:
        return False
    return True

def is_overlapping_rectangles(rect1, rect2):
    (t1, b1, l1, r1) = rect1
    (t2, b2, l2, r2) = rect2
    
    if r1 < l2 or r2 < l1: return False
    if b1 < t2 or b2 < t1: return False
    return True
