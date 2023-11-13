def prepare_data(self):
        """Called by Lightning trainer first.
        Prepares dataframe an splits it to training/validation/test as specified.
        """
        # Handle uncertain and missing labels
        print("prepare data")
        self.dataframe = preprocessing.handle_missing_labels(
            self.dataframe,
            self.hparams.labels,
            self.hparams.icd_refinement,
            self.hparams.bin_mapping,
        )

        # Handle view positions in dataframe
        self.dataframe = preprocessing.handle_view_positions(self.dataframe)
    
        # Decide between single- or multi-view
        if self.hparams.multi_view:
            # Each row contains both images from the study
            self.dataframe = preprocessing.to_multi_view(self.dataframe)
        else:
            # Reduce dataset to only frontal perspective
            self.dataframe = self.dataframe[self.dataframe.view_position == "FRONTAL"]

        # Process Demographic features
        if self.demog_features:
            self.dataframe = preprocessing.process_demogs(self.dataframe)

        # Remove all examples with missing modalities
        if self.hparams.clico_dropna:
            self.dataframe = self.dataframe.dropna(axis=0, subset=self.clico_features, how="all")
            self.dataframe.reset_index(drop=True, inplace=True)

        # Split train, test, val
        self.split = preprocessing.split_without_overlap(
            self.dataframe,
            self.hparams.random_state,
            self.hparams.split_fracs,
        )
        # Process Clico Features, if clico features are provided
        if self.clico_features:
           
            self.split = preprocessing.process_clicos(
                self.split,
                self.clico_features,
                self.hparams.random_state,
                self.hparams.clico_transform,
                #missing_modalities=not self.hparams.clico_dropna,
                missing_modalities=False,
            )

def setup(self):
        """Called by Lightning after prepare_data.
        Creates a dataset for each element of the split created in prepare_data.
        """
        self.datasets = {}
        print("stage")
        for key in self.split.keys():
            self.datasets[key] = MIMIC_CXR(
                self.split[key],
                labels=self.hparams.labels,
                clico_features=self.clico_features,
                clino_features=self.clino_features,
                demog_features=self.demog_features,
                local_root=self.local_root,
                multi_view=self.hparams.multi_view,
            )