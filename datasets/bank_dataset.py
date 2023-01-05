from aif360.datasets import StandardDataset


class BankDataset(StandardDataset):
    """Bank marketing Dataset.
    See :file:`aif360/data/raw/bank/README.md`.
    """

    def __init__(
        self,
        df_data,
        label_name="y",
        favorable_classes=["yes"],
        protected_attribute_names=["age"],
        privileged_classes=[lambda x: x >= 25],
        instance_weights_name=None,
        categorical_features=["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"],
        features_to_keep=[],
        features_to_drop=[],
        na_values=["unknown"],
        custom_preprocessing=None,
        metadata=None,
    ):

        super(BankDataset, self).__init__(
            df=df_data,
            label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing,
            metadata=metadata,
        )
