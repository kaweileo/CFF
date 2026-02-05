**Read this in other languages: [English](README.md), [中文](README_zh.md).**

# CFF

# Tomato Leaf Chlorophyll SPAD Value Time Series Dataset
Open Sourced Based on Postgraduate Thesis Data (Only Chlorophyll SPAD Value Time Series-Related Content Retained)
> Note: This dataset only contains chlorophyll SPAD value time series, provided by the research team of Professor Kaibin Wei and funded by the Gansu Provincial Natural Science Foundation. Users shall abide by the MIT License and unauthorized commercial use is prohibited.

## I. Dataset Construction Background and Methods
### 1.1 Analysis of Tomato Growth Cycle Characteristics
The tomato growth cycle is divided into seed germination, seedling growth, vegetative growth, flowering, fruit expansion, ripening and other stages. This study focuses on the dynamic changes of leaf chlorophyll content (characterized by SPAD values) at different growth stages, reveals its correlation mechanism with growth regulation and stress response, and provides data support for the precise management of protected agriculture.

### 1.2 Experimental Design and Material Preparation
- **Experimental Materials**: The tomato cultivar *Fenhongguan* was selected as the research object. This cultivar has the advantages of thick leaves, stable growth and constant chlorophyll content, which is superior to other cultivars such as strawberry tomato.
- **Instruments and Equipment**: The IN-YL01 Chlorophyll Analyzer was used to measure SPAD values, with an accuracy of ±2 SPAD units (range: 0-250). Regular calibration was conducted to ensure data reliability.
- **Cultivation Conditions**: Pot cultivation mode was adopted with one plant per pot, and the planting density was controlled to optimize ventilation and light transmission. A compound microbial fertilizer was applied, and stable growth environment was maintained in combination with irrigation, shading and wind protection measures.

### 1.3 Data Collection Methods and Procedures
#### Sampling Protocol
- Sample size: 20 plants × 2 leaves × 90 days = 3,600 samples. Sampling was conducted daily from 09:00 to 11:00 (avoiding the peak of photosynthetic fluctuation).
- Spatial positioning: Mature functional leaves at the 3rd to 5th nodes of the main stem were selected to avoid interference from young leaves and senescent leaves.
- Measurement replication: Three sampling points were set on each leaf (5±0.5 mm from the leaf vein) with three replications, and the coefficient of variation (CV) was controlled to be less than 5%.
- Standardized procedure: The baseline correction method was adopted to eliminate the influence of environmental fluctuations, with the formula as follows:
$$\text{Standardized SPAD Value} = \frac{\text{Measured Value} - \text{Daily Mean of Control Group}}{\text{Daily Standard Deviation of Control Group}}$$
A total of 3,199 valid samples were retained after quality control.

### 1.4 Data Processing and Dataset Construction
#### Abnormal data processing
- A review process was initiated when the difference rate of adjacent measured values exceeded 15%. Missing values were supplemented by mean imputation of adjacent plants or eliminated by row deletion.
- The Mean-Range Control Chart was used to monitor data stability (e.g., Plant No.9 and No.11).
#### Dataset composition
The dataset contains spatiotemporal series data of 20 plants × 2 leaves × 3 measurement sites (leaf apex/leaf middle/leaf base) over 30 days, with fields including Plant ID, Leaf Position, Measurement Time, SPAD Value and so on.

## II. Dataset Links and Descriptions
### 2.1 Self-Constructed Dataset (Tomato Chlorophyll SPAD Value Time Series)
- File: tomato_plus_8H.zip
- Link: [Baidu Netdisk](https://pan.baidu.com/s/1K9Pj_A1YMlCiluWfhhQGvA?pwd=uasm)
- Extraction Code: uasm
- Content: Contains a CSV format data file recording the SPAD value time series of tomato plants over 90 days, including sample metadata (Plant ID, Leaf Position, Sampling Time, etc.).

### 2.2 Public Datasets (Cited for Reference)
- File: Public Datasets.zip
- Link: [Baidu Netdisk]((https://pan.baidu.com/s/1N7rPSpunJQIr2cRWxxqsLg?pwd=3dhd)
- Extraction Code: 3dhd
- Link: [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)

The following multi-domain time series datasets are provided for comparative analysis:
- Weather3: 21 meteorological indicators (humidity, air temperature, etc.) in Germany, [Link](https://www.bgc-jena.mpg.de/wetter/)
- Traffic4: Road occupancy rate recorded by highway sensors in San Francisco, [Link](https://pems.dot.ca.gov/)
- Electricity5: Hourly electricity consumption of 321 households, [UCI Dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
- ILI6: Weekly number and proportion of influenza-like illness (ILI) cases, [CDC Data Platform](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html)
- ETT7: Power Transformer Temperature Dataset, [GitHub Repository](https://github.com/zhouhaoyi/ETDataset)
- Exchange-rate8: Daily exchange rates of 8 national currencies (use cautiously as a prediction benchmark), [GitHub Repository](https://github.com/laiguokun/multivariate-time-series-data)

## III. Acknowledgements and Support
### 3.1 Contributors
Special thanks to the following individuals for their technical support:
- @Professor Kaibin Wei (Supervisor) — Research guidance and resource support
- **@zhangyunjin488**
- **@jjqbxm**
- **@gqzszzy**
- **@Precious375**
- **@STTT1248**

### 3.2 Funding and Cooperation
- Funding Support: Gansu Provincial Natural Science Foundation
- Cooperating Institutions: Tianshui Normal University, Northwest Normal University

## IV. Open Source License and Citation
### 4.1 License Agreement
This dataset is licensed under the MIT License, which permits academic research and non-commercial use on the condition that the original attribution and citation are retained.

### 4.2 Citation Format
```bibtex
@dataset{tomato_leaf_image_dataset,  
  title={Tomato Leaf Chlorophyll SPAD Value Time Series Dataset},  
  author={kaweileo},  
  year={2025},  
  url={https://github.com/kaweileo/CFF}  
}
```
> Note: The duplicate `https://` in the original URL has been corrected to comply with URI specifications.

## V. Contact Information
For technical support or authorization for data citation, please contact us via GitHub Issue.
