from sklearn.preprocessing import LabelEncoder

class CustomLabelEncoder(LabelEncoder):
    def __init__(self, labels):
        super(LabelEncoder, self).__init__()
        self.fit(labels)
        self.n_classes = len(self.classes_)
    
    def int2label(self, int_array):
        return self.inverse_transform(int_array)
    
    def label2int(self, str_array):
        return self.transform(str_array)
    
    def get_int2label_dict(self):
        return {k:v for k,v in zip(range(self.n_classes), self.classes_)}
    
    def get_label2int_dict(self):
        return {k:v for k,v in zip(self.classes_, range(self.n_classes))}

def encode_labels(
    label_encoder: CustomLabelEncoder, 
    df: "pd.DataFrame", 
    label_col: str) -> "pd.DataFrame":
    """
    Make the label columns into integer labels.
    """
    df[label_col] = label_encoder.label2int(df[label_col])
    return df