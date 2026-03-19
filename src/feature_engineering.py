def compute_features(raw_input):

    username = raw_input["username"]
    fullname = raw_input["fullname"]
    bio = raw_input["bio"]
    
    # for nums/length username feature
    nums_length_username = sum(c.isdigit() for c in username) / len(username)
    
    # for nums/length fullname feature
    nums_length_fullname = sum(c.isdigit() for c in fullname) / len(fullname)
    
    # for fullname words feature
    fullname_words = len(fullname.split())
    
    # for name==username feature
    name_equal_username = int(username.lower() == fullname.lower())
    
    # for description length feature
    description_length = len(bio)

    features = {
        "profile pic": raw_input["profile_pic"],
        "nums/length username": nums_length_username,
        "fullname words": fullname_words,
        "nums/length fullname": nums_length_fullname,
        "name==username": name_equal_username,
        "description length": description_length,
        "external URL": raw_input["external_url"],
        "private": raw_input["private"],
        "#posts": raw_input["posts"],
        "#followers": raw_input["followers"],
        "#follows": raw_input["follows"]
    }

    return features