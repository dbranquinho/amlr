import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import figure
import sys
import datetime
import pkg_resources
import shutil
from PIL import Image
import io
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2ODeepLearningEstimator
import cv2
from pycm import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class report:

    def binary_class(self, type, target, 
                     duplicated, sep, exclude):

        img = plt.figure()
        self.write_image(img,'blank',width=600,height=500)
        
        self.gstep(0, "Reading Dataset")
        
        buffer = io.StringIO()
        self.dfo.columns = [c.replace(' ', '_') for c in self.dfo.columns]

        self.gstep(1, "Verify if duplicated")
        self.insert_text("shape", str(self.dfo.shape[0]) + ' / ' + str(self.dfo.shape[1]))
        self.get_classes(self.dfo, target)
        self.insert_text("nclasses", str(self.nclasses))
        self.insert_text("allclasses", str(self.allclasses))
        shape_before = self.dfo.shape[0]
        if duplicated:
            self.dfo = self.dfo.drop_duplicates(self.dfo.columns)
            shape_after = self.dfo.shape[0]
        if shape_before == shape_after:
            self.insert_text("duplicated", "none")
        else:
            self.insert_text("duplicated", str(shape_after - shape_before))
        
        if exclude != 'none':
            self.dfo.drop(columns=exclude, inplace=True)

        self.gstep(1, "Detecting hi frequency features")
        exclude = self.hi_freq(self.dfo)
        self.dfo.drop(columns=exclude['Feature'], inplace=True)
        
        hi_freq = self.w_table(data=exclude, border=0, align='left', 
                       collapse='collapse', color='black', 
                       foot=False)
        self.insert_text("excluded", hi_freq)

        self.gstep(1, "Encoding as sort_by_response")
        self.dfo_encode = self.encode(self.dfo.copy())
        
        self.gstep(1, "Basic Informations")
        
        df_info = pd.DataFrame()
        for column in self.dfo.columns:
            not_null = int(self.dfo.shape[0] - int(self.dfo[column].isna().sum()))
            dtype = self.dfo[column].dtypes
            df_info = df_info.append({'column': column, 'not_null': not_null, 'dtype': dtype}, ignore_index=True)
        df_info['not_null'] = df_info['not_null'].apply(lambda x: int(x))
        df_info['percent'] = df_info['not_null'].apply(lambda x: float("{:.4f}".format(1-(x/self.dfo.shape[0]))))
        info_dataset = self.w_table(data=df_info, border=0, align='left', 
                       collapse='collapse', color='black', 
                       foot=False)
        self.insert_text("info_dataset", info_dataset)

        self.gstep(1, "Computing Regression")

        Y = self.dfo_encode[target]
        dfo_num = self.dfo_encode[self.dfo_encode._get_numeric_data().columns]
        X = dfo_num.drop(columns=[target])


        # Criando os dados de train e test
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        cols = X.columns
        formule = " + ".join(map(str, cols))
        formule = target + " ~ " + formule
        reg = smf.ols(formule, data = dfo_num)
        res = reg.fit()
        self.insert_text('regression',str(res.summary()))

        self.gstep(1, "Unbalance Classes")

        temp = self.dfo[target].value_counts()
        df = pd.DataFrame({target: temp.index,'values': temp.values})
        plt.figure(figsize = (6,6))
        plt.title('Data Set - target value - data unbalance\n (' + target + ')')
        sns.set_color_codes("pastel")
        sns.barplot(x = target, y="values", data=df)
        locs, labels = plt.xticks()
        self.write_image(plt, "unbalance", width=500, height=350,crop=True)
        
        self.gstep(1, "Correlation")

        plt.clf()
        corr = self.dfo_encode.corr()
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        plt.figure(figsize=(8, 8))
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1,center=0, annot=True,
                    square=True, linewidths=1.5, cbar_kws={"shrink": .5})
        self.write_image(plt, "corr", width=0, height=0,crop=True)
        
        self.gstep(1, "Detecting Multicollinearity with VIF")

        y = self.dfo_encode[target]
        y = y.apply(lambda x: 1 if x == 'yes' else 0) 
        X = self.dfo_encode.drop(target, axis=1)
        X = X[X._get_numeric_data().columns]
        X = X.fillna(0) 
        X = X.dropna()
        vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        cols = X.columns
        cols = cols[cols!=target]
        df_m = pd.DataFrame({'cols': cols, 'vif': vif })
        df_m['significant'] = ''
        df_m['significant'] = df_m['vif'].apply(self.parse_values)
        m_vif = self.w_table(data=df_m, border=0, align='left', 
                       collapse='collapse', color='black', 
                       foot=False)        
        self.insert_text("vif", str(m_vif))

        i = 2
        text = ''
        text2 = ''
        for column in self.dfo.columns:
            feature = self.dfo[column].describe()
            text = text + '<option value="' + str(i) + '"> ' + column + ' </option>n\t\t\t\t\t\t\t\t'
            text2 = text2 + "\n\t\t\t\t\t\t\t\t\t\t} else if (selectedValue == '" + str(i) + "') {\n\t\t\t\t\t\t\t\tdivElement.innerHTML = '" + pd.DataFrame(feature).to_html().replace('\n','') + "';\n\t\t\t\t\t\t\t\t"
            i = i + 1
        text2 = text2 + '\n\t\t\t\t\t\t\t\t};'            
        self.insert_text('vif_desc_option',text)
        self.insert_text('vif_desc_table',text2)

        self.gstep(1, "Residual Analisys")

        plt.clf()
        model = Ridge()
        visualizer = ResidualsPlot(model, hist=False, qqplot=True)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        self.write_image(plt, "residual1", width=500, height=350,crop=True)    
        plt.clf()
        visualizer = ResidualsPlot(model, hist=True, qqplot=False)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        self.write_image(plt, "residual2", width=500, height=350,crop=True)

        self.gstep(1, "Initializing H2O")
        h2o.init()
        self.gstep(1, "Parsing Data Frame")
        df = h2o.H2OFrame(self.dfo_encode)
        self.gstep(1, "Trainning Auto Machine Learning")
        train, valid, test = df.split_frame(ratios=[0.7, 0.2], seed=1234)
        x = train.columns
        y = target
        x.remove(y)
        train[y] = train[y].asfactor()
        test[y] = test[y].asfactor()
        aml = H2OAutoML(max_models=20, max_runtime_secs=10, 
                        seed=1, include_algos = ["GLM", "DeepLearning", "DRF","xGBoost","StackedEnsemble"],
                        balance_classes=True)
        aml.train(x=x, y=y, training_frame=train)

        lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')        
        lb = lb.as_data_frame()
        lb = lb.drop(columns=['rmse','mse','predict_time_per_row_ms'])
        text = self.w_table(lb)
        self.insert_text('auto_ml_results', text)
        self.write_image(aml.varimp_heatmap(),'var_imp_model',width=450,height=400,crop=True)

        self.gstep(1, "AML - Partial Dependence")

        i = 101
        text = ''
        text2 = ''
        for column in tqdm(self.dfo.columns):
            feature = self.dfo[column].describe()
            text = text + '<option value="' + str(i) + '"> ' + column + ' </option>n\t\t\t\t\t\t\t\t'
            text2 = text2 + "\n\t\t\t\t\t\t\t\t\t\t} else if (selectedValue2 == '" + str(i) + "'){\n\t\t\t\t\t\t\t\tdivElement2.innerHTML = '<img src=\"images/img_aml_pd_" + str(i) + ".png\">';\n\t\t\t\t\t\t\t\t"
            self.write_image(aml.pd_multi_plot(valid,column),'aml_pd_' + str(i),width=600,height=500)
            i = i + 1
        text2 = text2 + '\n\t\t\t\t\t\t\t\t};'            
        self.insert_text('aml_pd_option',text)
        self.insert_text('aml_pd_image',text2)

        self.gstep(1, "Trainning (GLM) Gradient Linear Model to Ensemble")

        nfolds = 5
        family="binomial"

        amlr_glm = H2OGeneralizedLinearEstimator(family=family,
                                            nfolds=nfolds,
                                            lambda_ = 0,
                                            balance_classes=True,
                                            fold_assignment="Modulo",
                                            compute_p_values = True,
                                            keep_cross_validation_predictions=True,
                                            remove_collinear_columns = True)
        amlr_glm.train(x, y, training_frame=train)

        self.gstep(1, "Trainning (DRF) Dynamic Random Forest to Ensemble")
        amlr_rf = H2ORandomForestEstimator(ntrees=50,
                                        nfolds=nfolds,
                                        fold_assignment="Modulo",
                                        balance_classes=True,
                                        keep_cross_validation_predictions=True,
                                        seed=1)
        amlr_rf.train(x=x, y=y, training_frame=train)

        self.gstep(1, "Trainning (GBM) Gradient Boost Estimator Model to Ensemble")
        amlr_gbm = H2OGradientBoostingEstimator(nfolds=nfolds,
                                            seed=1111,
                                            balance_classes=True,
                                            fold_assignment="Modulo",
                                            keep_cross_validation_predictions = True)
        amlr_gbm.train(x=x, y=y, training_frame=train)

        self.gstep(1, "Trainning xGBoost Model to Ensemble")
        amlr_xgb = H2OXGBoostEstimator(booster='dart',
                                    nfolds=nfolds,
                                    normalize_type="tree",
                                    fold_assignment="Modulo",
                                    keep_cross_validation_predictions=True,
                                    seed=1234)
        amlr_xgb.train(x=x,y=y, training_frame=train, validation_frame=valid)

        self.gstep(1, "Trainning Deep Learning Model to Ensemble")

        family="bernoulli"
        dl_model = H2ODeepLearningEstimator(distribution=family,
                                hidden=[1],
                                epochs=1000,
                                train_samples_per_iteration=-1,
                                reproducible=True,
                                activation="Tanh",
                                single_node_mode=False,
                                balance_classes=True,
                                force_load_balance=False,
                                seed=23123,
                                tweedie_power=1.5,
                                score_training_samples=0,
                                score_validation_samples=0,
                                stopping_rounds=0)
        dl_model.train(x=x, y=y, training_frame=train)

        self.gstep(1, "Trainning Ensemble")
        ensemble = H2OStackedEnsembleEstimator(model_id="amlr_ensemble",
                                            base_models=[amlr_gbm, amlr_rf, amlr_xgb, amlr_glm])
        ensemble.train(x=x, y=y, training_frame=train)

        i = 201
        text = ''
        text2 = ''
        self.gstep(1, "Ensamble - (ICE) Individual Condition Expectation")
        for column in tqdm(self.dfo.columns):
            feature = self.dfo[column].describe()
            text = text + '<option value="' + str(i) + '"> ' + column + ' </option>n\t\t\t\t\t\t\t\t'
            text2 = text2 + "\n\t\t\t\t\t\t\t\t\t\t} else if (selectedValue3 == '" + str(i) + "'){\n\t\t\t\t\t\t\t\tdivElement3.innerHTML = '<img src=\"images/img_ice_pd_" + str(i) + ".png\">';\n\t\t\t\t\t\t\t\t"
            self.write_image(ensemble.ice_plot(valid,column),'ice_pd_' + str(i),width=600,height=500)
            i = i + 1
        text2 = text2 + '\n\t\t\t\t\t\t\t\t};'            
        self.insert_text('ice_pd_option',text)
        self.insert_text('ice_pd_image',text2)

        self.gstep(1, "AMLR - Correlation by Model")
        self.write_image(aml.model_correlation_heatmap(test),'aml_correlation_models')

        self.gstep(1, "Processing Models Performance")

        i = 0
        dfp = pd.DataFrame({'Algo': []})
        outcome = list(valid[target].as_data_frame()[target])
        for algo in ['GLM','Random Forest','GBM','xGBoost','Deep Learning']:
            plt.clf()
            if algo == 'GLM':
                predict = list(amlr_glm.predict(valid).as_data_frame()['predict'])
                cf_table='cf_glm'
                cm_glm = ConfusionMatrix(outcome, predict)
                glm_var_imp = amlr_glm._model_json['output']['variable_importances'].as_data_frame()
                x = glm_var_imp['percentage']
                x.index = glm_var_imp['variable']
                x.sort_values().plot(kind='barh')
                plt.xlabel('Percentage')
                fig = plt.gcf()
                self.write_image(fig,'fi_glm',width=450,height=450)
                
                
            if algo == 'Random Forest':
                predict = list(amlr_rf.predict(valid).as_data_frame()['predict'])
                cf_table='cf_rf'
                cm_rf = ConfusionMatrix(outcome, predict)
                rf_var_imp = amlr_rf._model_json['output']['variable_importances'].as_data_frame()
                x = rf_var_imp['percentage']
                x.index = rf_var_imp['variable']
                x.sort_values().plot(kind='barh')
                plt.xlabel('Percentage')
                fig = plt.gcf()
                self.write_image(fig,'fi_rf',width=450,height=450)
            if algo == 'GBM':
                predict = list(amlr_gbm.predict(valid).as_data_frame()['predict'])
                cf_table='cf_gbm'
                cm_gbm = ConfusionMatrix(outcome, predict)
                gbm_var_imp = amlr_gbm._model_json['output']['variable_importances'].as_data_frame()
                x = gbm_var_imp['percentage']
                x.index = gbm_var_imp['variable']
                x.sort_values().plot(kind='barh')
                plt.xlabel('Percentage')
                fig = plt.gcf()
                self.write_image(fig,'fi_gbm',width=450,height=450)
            if algo == 'xGBoost':
                predict = list(amlr_xgb.predict(valid).as_data_frame()['predict'])
                cf_table='cf_xgb'
                cm_xgb = ConfusionMatrix(outcome, predict)
                xgb_var_imp = amlr_xgb._model_json['output']['variable_importances'].as_data_frame()
                x = xgb_var_imp['percentage']
                x.index = xgb_var_imp['variable']
                x.sort_values().plot(kind='barh')
                plt.xlabel('Percentage')
                fig = plt.gcf()
                self.write_image(fig,'fi_xgb',width=450,height=450)
            if algo == 'Deep Learning':
                predict = list(dl_model.predict(valid).as_data_frame()['predict'])
                cf_table='cf_dl'
                cm_dl = ConfusionMatrix(outcome, predict)
                dl_var_imp = dl_model._model_json['output']['variable_importances'].as_data_frame()
                x = dl_var_imp['percentage']
                x.index = dl_var_imp['variable']
                x.sort_values().plot(kind='barh')
                plt.xlabel('Percentage')
                fig = plt.gcf()
                self.write_image(fig,'fi_dl',width=450,height=450)
            # Confusion Matrix for all models
            cm = confusion_matrix(predict, outcome)
            cm = pd.DataFrame(cm)
            cr = classification_report(outcome, predict,target_names=self.allclasses,output_dict=True)
            table_cr = pd.DataFrame(cr).transpose().round(4)
            table_cr.reset_index(level=0, inplace=True)
            table_cr = table_cr.rename(columns={'index': 'Description'})
            table_model = self.w_table(data=table_cr, border=0, align='left', 
                                        collapse='collapse', color='black', 
                                        foot=False)        
            self.insert_text(cf_table, str(table_model))            

            # Statistcs for all metrics
            cm = ConfusionMatrix(outcome, predict)
            dfp = pd.concat([dfp, pd.DataFrame(cm.overall_stat)[1:]],ignore_index=True)
            dfp.loc[i:,['Algo']] = algo
            i = i + 1
        dfp = dfp.round(4)
    
        cp = Compare({'RF':cm_rf,'GLM':cm_glm,'GBM':cm_gbm,'XGB':cm_xgb,'DL':cm_dl})
        cp_best_name = cp.best_name
        cp = pd.DataFrame(cp.scores)
        cp.reset_index(level=0, inplace=True)
        cp = cp.rename(columns={'index': 'Description'})        
        table_cp = self.w_table(data=cp, border=0, align='left', 
                                    collapse='collapse', color='black', 
                                    foot=False)        
        if str(cp_best_name) == 'None':
            cp_best_name = 'Confusion matrices are too close and the best one can not be recognized.'
            max_v = cp.loc[0][1:].max()
            i = 0
            list_max = list()
            for column in cp.columns:
                if i > 0:
                    if cp[column][0] >= max_v:
                        list_max.append(column)
                i = i + 1
            self.insert_text("the_best_name", "Winners: " + ' - '.join(list_max) + '<br>' + cp_best_name)

        else:
            self.insert_text("the_best_name", str(cp_best_name))
        
        self.insert_text("best_algorithms", str(table_cp))
        self.insert_text("the_best_name", str(cp_best_name))
        
        table_model = self.w_table(data=dfp, border=0, align='left', 
                                    collapse='collapse', color='black', 
                                    foot=False)        
        self.insert_text("table_performance", str(table_model))

        self.gstep(1, "Closing!! All works are done!!")
        # write report        
        self.write_report(self.index_html)
        
        
    def multi_class():
        pass

    def __init__(self, output_report='report'):
        resource_package = __name__
        index_path = '/'.join(('templates', 'index.html'))
        css_path = '/'.join(('templates', 'index.css'))
        indexpath = pkg_resources.resource_filename(resource_package, index_path)
        csspath = pkg_resources.resource_filename(resource_package, css_path)

        resource_path = '/'.join(('templates', 'templates.dat'))
        filepath = pkg_resources.resource_filename(__name__, resource_path)
        self.df_template = pd.read_csv(filepath, sep=';')
                
        if not os.path.exists(output_report):
            os.mkdir(output_report)
        shutil.copy2(indexpath, output_report)
        shutil.copy2(csspath, output_report)
        self.index_html = ''
        self.load_report()
        plt.rc('figure', max_open_warning = 0)


    def init_h2o(self, progress=False):
        self.h2o.init()
        if not progress:
            self.h2o.no_progress()


    def get_classes(self, dataset='none', target='none'):
        self.nclasses = len(dataset[target].unique())
        self.allclasses = dataset[target].unique()
        if self.nclasses == 2:
            self.type_class = 'b'
        elif self.nclasses > 2:
            self.type_class = 'm'
        else:
            raise ValueError("Number of classes not permited %s" % self.nrclasses)

    def write_image(self, graph, file, width=800, height=600, crop=False):
        if not os.path.exists('report/images'):
            os.mkdir('report/images')  
        graph.savefig('report/images/img_' + file + '.png')
        if crop:
            self.crop_image('report/images/img_' + file + '.png')
        if width > 0:
            picture = Image.open('report/images/img_' + file + '.png')
            picture.thumbnail(size=(width,height))
            picture.save('report/images/img_' + file + '.png')
            picture.close()

    def load_report(self):
        text_file = open('report/index.html','r')
        Lines = text_file.readlines()
        str = ""
        for line in Lines:
            str = str + line
        text_file.close()
        self.index_html = str
    
    def write_report(self, text, mode='w+'):
        file2write = 'report/index.html'
        file = open(file2write,mode)
        file.write(text)
        file.close()        

    def insert_text(self, mask, text):
        self.index_html = self.index_html.replace('{{' + mask + '}}', text)

    def read_template(self, file):
        resource_package = __name__
        file_read = self.df_template[self.df_template['template'] == file]['file']
        file_read = file_read.to_string().split(' ')[-1]
        resource_path = '/'.join(('templates', file_read))
        filepath = pkg_resources.resource_filename(resource_package, resource_path)
        try:
            text_file = open(filepath,"r")
        except:
            raise ValueError('Template not found!!!')

        Lines = text_file.readlines()
        str = ""
        for line in Lines:
            str = str + line
        text_file.close()        
        return(str)


    def w_table(self, data="none", border=1, align='left', 
                      collapse='collapse', color='black', 
                      foot=False):

        if foot:
            sum_shape = -1
        else:
            sum_shape = 0

        if str(type(data)) != "<class 'str'>":
            self.tableh = self.read_template('tableh')
            self.tableh = self.tableh.replace('{{align}}', align)
            self.tableh = self.tableh.replace('{{collapse}}', collapse)
            self.tableh = self.tableh.replace('{{border_size}}', str(border))
            self.tableh = self.tableh.replace('{{border_color}}', color)
            self.tablef = self.read_template('tablef')
        else:
            raise ValueError('No dataframe provided to write table, I need one')

        table_tag = self.tableh
        self.theadh = self.read_template('table_theadh')
        self.theadb = self.read_template('table_theadb')
        self.theadf = self.read_template('table_theadf')
        self.tbodyh = self.read_template('table_tbodyh')
        self.tbodyb = self.read_template('table_tbodyb')
        self.tbodyf = self.read_template('table_tbodyf')

        # building table head 
        table_tag = table_tag + self.theadh
        for col_names in data.columns:
            self.field = self.theadb.replace('{{field}}', col_names)
            table_tag = table_tag + self.field
        table_tag = table_tag + self.theadf

        # building table body
        table_tag = table_tag + self.tbodyh
        for i in range(data.shape[0] + sum_shape):
            table_tag = table_tag + '<tr>\n'
            for j in range(data.shape[1]):
                self.field = self.tbodyb.replace('{{field}}', str(data.iloc[i][j]))
                table_tag = table_tag + self.field
            table_tag = table_tag + '</tr>\n'
        table_tag = table_tag + self.tbodyf

        # building table foot
        if foot:
            self.tfooth = self.read_template('table_tfooth')
            self.tfootb = self.read_template('table_tfootb')
            self.tfootf = self.read_template('table_tfootf')
            i = data.shape[0] -1
            table_tag = table_tag + self.tfooth
            for j in range(data.shape[1]):
                self.field = self.tfootb.replace('{{field}}', str(data.iloc[i][j]))
                table_tag = table_tag + self.field
            table_tag = table_tagy + self.tfootf
        else:
            table_tag = table_tag + self.tablef
        return(table_tag)


    def parse_values(self, x):
        if x < 1.5:
            return 'good'
        elif x >= 1.5 and x < 3:
            return 'moderate'
        elif x >=3 and x <10:
            return 'attention'
        else:
            return 'high'
            
    def encode(self, dfo):
        dfo = dfo.fillna(0)
        for column in tqdm(dfo.select_dtypes("object").columns):
            encode = pd.DataFrame(dfo[column].value_counts())
            encode = encode.reset_index()
            encode = encode.rename(columns={column: 'Freq'})
            encode.reset_index(drop=True, inplace=True)
            encode = encode.rename(columns={'index': column})
            ct = encode.shape[0]
            for i in range(ct):
                dfo[column] = dfo[column].apply(lambda x: ct -i if x == encode[column][i] else x)
        return(dfo)                

    def gstep(self, init=0, text='not defined'):
        end=24
        if init==0:
            self.step = 1
        self.step = self.step + init
        print("Step " + str(self.step) + "/" + str(end) + " - " + text)

    def crop_image(self, filename):
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bit = cv2.bitwise_not(gray)
        amtImage = cv2.adaptiveThreshold(bit, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 15)
        kernel = np.ones((15,15),np.uint8)
        erosion = cv2.erode(amtImage,kernel,iterations = 2)
        (height, width) = img.shape[0:2]
        image = erosion[50:height - 50, 50: width - 50]
        (nheight, nwidth) = image.shape[0:2]
        index = []
        for x in range (0, nheight):
            line = []
            for y in range(0, nwidth):
                line2 = []
                if (image[x, y] < 150):
                    line.append(image[x, y])
            if (len(line) / nwidth > 0.2):    
                index.append(x)
        index2 = []
        for a in range(0, nwidth):
            line2 = []
            for b in range(0, nheight):
                if image[b, a] < 150:
                    line2.append(image[b, a])
            if (len(line2) / nheight > 0.15):
                index2.append(a)
        img = img[min(index):max(index) + min(250, (height - max(index))* 10 // 11) , \
            max(0, min(index2)): max(index2) + min(250, (width - max(index2)) * 10 // 11)]
        cv2.imwrite(filename, img)

    def hi_freq(self, df):
        total = df.shape[0]
        exclude = pd.DataFrame({'Feature': [], 'Freq': []})
        for column in df.columns:
            nr = df[column].value_counts().count()
            if nr/total > 0.5:
                exclude = exclude.append({'Feature': column, 'Freq': nr/total}, ignore_index=True)
        return exclude
                        
    def create_report(self, dataset='none', data_frame='none',
                      type='html', target='none', 
                      duplicated=True, sep=';', exclude='none'):
        
        if dataset == 'none':
            self.dfo = data_frame.copy()
        if target == 'none':
            raise ValueError("Target not defined")
        if type != 'html':
            raise ValueError("Report type not supported")
        
        if dataset != 'none':
            self.dfo = pd.read_csv(dataset, sep=sep)
            
        self.get_classes(self.dfo, target)
        if self.type_class == "b":
            self.binary_class(type, target, 
                      duplicated, sep, exclude)
        else:
            raise Exception("Multi Label Classification detected, not allowed for while")

