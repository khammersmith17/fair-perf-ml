use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum ModelType {
    LinearRegression,
    LogisticRegression,
    BinaryClassification,
}

impl TryFrom<&str> for ModelType {
    type Error = String;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "LinearRegression" => Ok(Self::LinearRegression),
            "LogisticRegression" => Ok(Self::LogisticRegression),
            "BinaryClassification" => Ok(Self::BinaryClassification),
            _ => Err("invalid model type".into()),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_from_str() {
        assert_eq!(
            ModelType::try_from("LinearRegression").unwrap(),
            ModelType::LinearRegression,
        );
        assert_eq!(
            ModelType::try_from("LogisticRegression").unwrap(),
            ModelType::LogisticRegression,
        );
        assert_eq!(
            ModelType::try_from("BinaryClassification").unwrap(),
            ModelType::BinaryClassification,
        );
    }
}
