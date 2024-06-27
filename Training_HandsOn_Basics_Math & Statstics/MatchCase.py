print('Hi')

ch = int(input("Enter a num: "))

match ch:
    case 1:
        print('One')
    case 2:
        print('Two')
    case _:
        print('Number not recognized')
