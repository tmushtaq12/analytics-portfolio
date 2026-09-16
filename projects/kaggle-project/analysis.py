from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'titanic.csv'
IMAGE_DIR = BASE_DIR / 'images'
IMAGE_DIR.mkdir(exist_ok=True)


def load_and_clean_data():
	"""Load the Kaggle dataset and apply reproducible cleaning rules."""
	passengers = pd.read_csv(DATA_PATH)
	passengers['Age'] = passengers['Age'].fillna(passengers['Age'].median())
	passengers['Embarked'] = passengers['Embarked'].fillna(passengers['Embarked'].mode()[0])
	passengers['FamilySize'] = passengers['SibSp'] + passengers['Parch'] + 1
	passengers['IsAlone'] = (passengers['FamilySize'] == 1).astype(int)
	passengers['AgeGroup'] = pd.cut(
		passengers['Age'],
		bins=[0, 12, 18, 35, 60, 100],
		labels=['Child', 'Teenager', 'Young adult', 'Adult', 'Senior'],
		include_lowest=True,
	)
	return passengers


def save_charts(passengers):
	"""Create portfolio-ready charts from the cleaned passenger data."""
	survival_by_sex = passengers.groupby('Sex', as_index=False)['Survived'].mean()
	survival_by_class = passengers.groupby('Pclass', as_index=False)['Survived'].mean()

	sns.set_theme(style='whitegrid')

	plt.figure(figsize=(8, 5))
	sns.barplot(data=survival_by_sex, x='Sex', y='Survived', hue='Sex', legend=False, palette='Set2')
	plt.title('Survival Rate by Sex')
	plt.ylabel('Survival rate')
	plt.ylim(0, 1)
	plt.tight_layout()
	plt.savefig(IMAGE_DIR / 'titanic_survival_by_sex.png', dpi=160)
	plt.close()

	plt.figure(figsize=(8, 5))
	sns.barplot(data=survival_by_class, x='Pclass', y='Survived', hue='Pclass', legend=False, palette='viridis')
	plt.title('Survival Rate by Passenger Class')
	plt.xlabel('Passenger class')
	plt.ylabel('Survival rate')
	plt.ylim(0, 1)
	plt.tight_layout()
	plt.savefig(IMAGE_DIR / 'titanic_survival_by_class.png', dpi=160)
	plt.close()

	plt.figure(figsize=(9, 5))
	sns.countplot(data=passengers, x='AgeGroup', hue='Survived', palette='Set1')
	plt.title('Passenger Outcomes by Age Group')
	plt.xlabel('Age group')
	plt.ylabel('Passenger count')
	plt.legend(title='Survived', labels=['No', 'Yes'])
	plt.tight_layout()
	plt.savefig(IMAGE_DIR / 'titanic_outcomes_by_age_group.png', dpi=160)
	plt.close()


def main():
	passengers = load_and_clean_data()
	survival_by_sex = passengers.groupby('Sex')['Survived'].mean().sort_values(ascending=False)
	survival_by_class = passengers.groupby('Pclass')['Survived'].mean().sort_values(ascending=False)

	print('Kaggle Titanic Survival Analysis')
	print('=' * 60)
	print(f'Records: {len(passengers):,}')
	print(f'Overall survival rate: {passengers["Survived"].mean():.1%}')
	print(f'Missing ages after cleaning: {passengers["Age"].isna().sum()}')
	print('\nSurvival rate by sex:')
	print((survival_by_sex * 100).round(1).astype(str).add('%').to_string())
	print('\nSurvival rate by passenger class:')
	print((survival_by_class * 100).round(1).astype(str).add('%').to_string())
	print('\nFamily-size finding:')
	print(passengers.groupby('IsAlone')['Survived'].mean().rename({0: 'Travelled with family', 1: 'Travelled alone'}).to_string())

	save_charts(passengers)
	print('\nCharts saved:')
	for chart in ['titanic_survival_by_sex.png', 'titanic_survival_by_class.png', 'titanic_outcomes_by_age_group.png']:
		print(f'- {IMAGE_DIR / chart}')


if __name__ == '__main__':
	main()
