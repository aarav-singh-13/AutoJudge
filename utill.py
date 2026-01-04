def score_to_class(score):
    if score <= 3.33:
        return 'easy'
    elif score >= 6.66:
        return 'hard'
    else:
        return 'medium'