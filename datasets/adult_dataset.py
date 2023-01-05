from aif360.datasets import StandardDataset


default_mappings = {
    "label_maps": [{1.0: ">50K", 0.0: "<=50K"}],
    "protected_attribute_maps": [{1.0: "White", 0.0: "Non-white"}, {1.0: "Male", 0.0: "Female"}],
}


class AdultDataset(StandardDataset):
    """Adult Census Income Dataset.
    See :file:`aif360/data/raw/adult/README.md`.
    """

    def __init__(
        self,
        df_data,
        label_name="income-per-year",
        favorable_classes=[">50K", ">50K."],
        protected_attribute_names=["race", "sex"],
        privileged_classes=[["White"], ["Male"]],
        instance_weights_name=None,
        categorical_features=["workclass", "education", "marital-status", "occupation", "relationship", "native-country"],
        features_to_keep=[],
        features_to_drop=["fnlwgt"],
        na_values=["?"],
        custom_preprocessing=None,
        metadata=default_mappings,
    ):

        super(AdultDataset, self).__init__(
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
