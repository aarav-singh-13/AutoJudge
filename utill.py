def score_to_class(score):
    if score <= 3.3:
        return 'easy'
    elif score >= 6.6:
        return 'hard'
    else:
        return 'medium'