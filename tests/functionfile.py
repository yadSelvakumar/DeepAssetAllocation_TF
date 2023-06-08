
def format_data_for_display(people_data):
    """Format the data for display."""
    return [
        f"{person['given_name']} {person['family_name']}: {person['title']}"
        for person in people_data
    ]
