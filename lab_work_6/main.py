def read_file(filename):
    result = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            surname = parts[0]
            place = None
            if len(parts) > 1:
                place = int(parts[1])
            
            result[surname] = place
    return result

def process_olympiad_data(file1, file2):
    olympiad1 = read_file(file1)
    olympiad2 = read_file(file2)
    
    all_students = set(olympiad1.keys()) | set(olympiad2.keys())
    
    results = []

    for student in sorted(all_students):
        place1 = olympiad1.get(student)
        place2 = olympiad2.get(student)
        
        if (student in olympiad1 and student not in olympiad2):
            if place1 is None:
                results.append(f"{student} д")

        elif (student not in olympiad1 and student in olympiad2):
            if place2 is None:
                results.append(f"{student} д")

        else:
            prize_count = 0

            if place1 is not None and 1 <= place1 <= 3:
                prize_count += 1

            if place2 is not None and 1 <= place2 <= 3:
                prize_count += 1

            results.append(f"{student} {prize_count + 3}")
    
    for result in results:
        print(result)

def main():
    process_olympiad_data('olympiad1.txt', 'olympiad2.txt')

if __name__ == "__main__":
    main()
