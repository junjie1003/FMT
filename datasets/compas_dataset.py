from aif360.datasets import StandardDataset


default_mappings = {
    "label_maps": [{1.0: "Did recid.", 0.0: "No recid."}],
    "protected_attribute_maps": [{0.0: "Male", 1.0: "Female"}, {1.0: "Caucasian", 0.0: "Not Caucasian"}],
}


class CompasDataset(StandardDataset):
    """ProPublica COMPAS Dataset.
    See :file:`aif360/data/raw/compas/README.md`.
    """

    def __init__(
        self,
        df_data,
        label_name="two_year_recid",
        favorable_classes=[0],
        protected_attribute_names=["sex", "race"],
        privileged_classes=[["Female"], ["Caucasian"]],
        instance_weights_name=None,
        categorical_features=["age_cat", "c_charge_degree", "c_charge_desc"],
        features_to_keep=[
            "sex",
            "age",
            "age_cat",
            "race",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "priors_count",
            "c_charge_degree",
            "c_charge_desc",
            "two_year_recid",
        ],
        features_to_drop=[],
        na_values=[],
        custom_preprocessing=None,
        metadata=default_mappings,
    ):

        super(CompasDataset, self).__init__(
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
