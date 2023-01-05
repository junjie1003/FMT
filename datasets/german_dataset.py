from aif360.datasets import StandardDataset


default_mappings = {
    "label_maps": [{1.0: "Good Credit", 2.0: "Bad Credit"}],
    "protected_attribute_maps": [{1.0: "Male", 0.0: "Female"}, {1.0: "Old", 0.0: "Young"}],
}


class GermanDataset(StandardDataset):
    """German credit Dataset.
    See :file:`aif360/data/raw/german/README.md`.
    """

    def __init__(
        self,
        df_data,
        label_name="credit",
        favorable_classes=[1],
        protected_attribute_names=["sex", "age"],
        privileged_classes=[["male"], lambda x: x > 25],
        instance_weights_name=None,
        categorical_features=[
            "status",
            "credit_history",
            "purpose",
            "savings",
            "employment",
            "other_debtors",
            "property",
            "installment_plans",
            "housing",
            "skill_level",
            "telephone",
            "foreign_worker",
        ],
        features_to_keep=[],
        features_to_drop=["personal_status"],
        na_values=[],
        custom_preprocessing=None,
        metadata=default_mappings,
    ):

        super(GermanDataset, self).__init__(
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
