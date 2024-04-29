import os
import time

def change_all_files_modified_date(directory, new_modified_time):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.utime(file_path, (new_modified_time, new_modified_time))

# Example usage
directory_path = r"S:\2021 Projects\ML Projects\Self_Driving_Car\final\test_videos"
new_modified_time = time.time()  # You can set this to the desired modified time
change_all_files_modified_date(directory_path, new_modified_time)
