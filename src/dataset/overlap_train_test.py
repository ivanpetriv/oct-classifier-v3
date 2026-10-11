import os


def get_patients(directory):
    patients = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if "-" in file:
                # Extracts the patient ID from format: CLASS-PATIENT_ID-ORDER
                patients.add(file.split("-")[1])
    return patients


train_patients = get_patients(
    "/run/media/ivan/01DCA2451BEF4120/LPNU/University/masters/Thesis/Ravelli/train"
)
test_patients = get_patients(
    "/run/media/ivan/01DCA2451BEF4120/LPNU/University/masters/Thesis/Ravelli/test"
)

overlap = train_patients.intersection(test_patients)
print(overlap)

if len(overlap) == 0:
    print("No overlap")
else:
    print(f"There are {len(overlap)} patients in both train and test sets.")
