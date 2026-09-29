{
  "kind": "specification_oracle_check",
  "created_utc": "2026-09-29T21:18:05.417551+00:00",
  "product_tests_executed": false,
  "result": "PASS",
  "case_count": 85,
  "errors": [],
  "checks": [
    {
      "name": "metric parser ORACLE-001",
      "result": "PASS",
      "actual": {
        "parsed_intent": "billing_refund",
        "exact_intent_match": 1,
        "response_schema_valid": 1
      },
      "expected": {
        "parsed_intent": "billing_refund",
        "exact_intent_match": 1,
        "response_schema_valid": 1
      }
    },
    {
      "name": "metric parser ORACLE-002",
      "result": "PASS",
      "actual": {
        "parsed_intent": "technical_issue",
        "exact_intent_match": 0,
        "response_schema_valid": 1
      },
      "expected": {
        "parsed_intent": "technical_issue",
        "exact_intent_match": 0,
        "response_schema_valid": 1
      }
    },
    {
      "name": "metric parser ORACLE-003",
      "result": "PASS",
      "actual": {
        "parsed_intent": null,
        "exact_intent_match": 0,
        "response_schema_valid": 0
      },
      "expected": {
        "parsed_intent": null,
        "exact_intent_match": 0,
        "response_schema_valid": 0
      }
    },
    {
      "name": "metric aggregate exact_intent_match",
      "result": "PASS",
      "actual": {
        "numerator": 1,
        "denominator": 3,
        "ratio": 0.3333333333333333
      },
      "expected": {
        "numerator": 1,
        "denominator": 3,
        "ratio": 0.3333333333333333
      }
    },
    {
      "name": "metric aggregate response_schema_valid",
      "result": "PASS",
      "actual": {
        "numerator": 2,
        "denominator": 3,
        "ratio": 0.6666666666666666
      },
      "expected": {
        "numerator": 2,
        "denominator": 3,
        "ratio": 0.6666666666666666
      }
    },
    {
      "name": "retrieval RQ01",
      "result": "PASS",
      "actual": [
        "RM01",
        "RM02",
        "RM03"
      ],
      "expected": [
        "RM01",
        "RM02",
        "RM03"
      ]
    },
    {
      "name": "retrieval score RQ01 RM01",
      "result": "PASS",
      "actual": 0.5044368928581016,
      "expected": 0.5044368928581014,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval score RQ01 RM02",
      "result": "PASS",
      "actual": 0.0,
      "expected": 0.0,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval score RQ01 RM03",
      "result": "PASS",
      "actual": 0.0,
      "expected": 0.0,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval RQ02",
      "result": "PASS",
      "actual": [
        "RM02",
        "RM04",
        "RM07"
      ],
      "expected": [
        "RM02",
        "RM04",
        "RM07"
      ]
    },
    {
      "name": "retrieval score RQ02 RM02",
      "result": "PASS",
      "actual": 0.34655096811857145,
      "expected": 0.3465509681185714,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval score RQ02 RM04",
      "result": "PASS",
      "actual": 0.27720198880720437,
      "expected": 0.27720198880720437,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval score RQ02 RM07",
      "result": "PASS",
      "actual": 0.06261552473805203,
      "expected": 0.06261552473805203,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval RQ03",
      "result": "PASS",
      "actual": [
        "RM03",
        "RM06",
        "RM05"
      ],
      "expected": [
        "RM03",
        "RM06",
        "RM05"
      ]
    },
    {
      "name": "retrieval score RQ03 RM03",
      "result": "PASS",
      "actual": 0.5329571614592222,
      "expected": 0.5329571614592222,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval score RQ03 RM06",
      "result": "PASS",
      "actual": 0.11350474029873131,
      "expected": 0.11350474029873134,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval score RQ03 RM05",
      "result": "PASS",
      "actual": 0.08247794972081855,
      "expected": 0.08247794972081855,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval RQ04",
      "result": "PASS",
      "actual": [
        "RM04",
        "RM01",
        "RM02"
      ],
      "expected": [
        "RM04",
        "RM01",
        "RM02"
      ]
    },
    {
      "name": "retrieval score RQ04 RM04",
      "result": "PASS",
      "actual": 0.46350816098461767,
      "expected": 0.46350816098461767,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval score RQ04 RM01",
      "result": "PASS",
      "actual": 0.0,
      "expected": 0.0,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval score RQ04 RM02",
      "result": "PASS",
      "actual": 0.0,
      "expected": 0.0,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval RQ05",
      "result": "PASS",
      "actual": [
        "RM05",
        "RM02",
        "RM04"
      ],
      "expected": [
        "RM05",
        "RM02",
        "RM04"
      ]
    },
    {
      "name": "retrieval score RQ05 RM05",
      "result": "PASS",
      "actual": 0.5910418222222802,
      "expected": 0.5910418222222802,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval score RQ05 RM02",
      "result": "PASS",
      "actual": 0.145066531654174,
      "expected": 0.14506653165417394,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval score RQ05 RM04",
      "result": "PASS",
      "actual": 0.13797832500708695,
      "expected": 0.13797832500708695,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval RQ06",
      "result": "PASS",
      "actual": [
        "RM06",
        "RM04",
        "RM07"
      ],
      "expected": [
        "RM06",
        "RM04",
        "RM07"
      ]
    },
    {
      "name": "retrieval score RQ06 RM06",
      "result": "PASS",
      "actual": 0.5619882961489768,
      "expected": 0.5619882961489769,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval score RQ06 RM04",
      "result": "PASS",
      "actual": 0.14498942582688443,
      "expected": 0.14498942582688443,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "retrieval score RQ06 RM07",
      "result": "PASS",
      "actual": 0.10683960606880566,
      "expected": 0.10683960606880566,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "LoRA unique trainable scalar count",
      "result": "PASS",
      "actual": 460800,
      "expected": 460800
    },
    {
      "name": "Tiny standard parameter count",
      "result": "PASS",
      "actual": 137088,
      "expected": 137088
    },
    {
      "name": "Tiny tied delta",
      "result": "PASS",
      "actual": 16448,
      "expected": 16448
    },
    {
      "name": "mean of three illustrative results",
      "result": "PASS",
      "actual": 1.3,
      "expected": 1.3,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "sample standard deviation",
      "result": "PASS",
      "actual": 0.09999999999999998,
      "expected": 0.1,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "byte aggregate is weighted",
      "result": "PASS",
      "actual": 0.9090909090909091,
      "expected": 0.9090909090909091,
      "absolute_tolerance": 1e-12
    },
    {
      "name": "materialized digest applied-intents-v1/test.jsonl",
      "result": "PASS",
      "actual": "53b1f2c077aa6d1e3fa2682504c478bc9c6ebe0d655b4fc0e139d3c1fb00845d",
      "expected": "53b1f2c077aa6d1e3fa2682504c478bc9c6ebe0d655b4fc0e139d3c1fb00845d"
    },
    {
      "name": "all SFT target formats applied-intents-v1/test.jsonl",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "materialized digest applied-intents-v1/train.jsonl",
      "result": "PASS",
      "actual": "987d32cc2ee7eade5b860ea4a24ab9dc6024f0b89ba5c61afc9c32291de6f3dc",
      "expected": "987d32cc2ee7eade5b860ea4a24ab9dc6024f0b89ba5c61afc9c32291de6f3dc"
    },
    {
      "name": "all SFT target formats applied-intents-v1/train.jsonl",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "materialized digest applied-intents-v1/validation.jsonl",
      "result": "PASS",
      "actual": "f5faa58929a13b374d55265456227de3ff01b2bda38ec9f157e78d36ee4710a5",
      "expected": "f5faa58929a13b374d55265456227de3ff01b2bda38ec9f157e78d36ee4710a5"
    },
    {
      "name": "all SFT target formats applied-intents-v1/validation.jsonl",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "materialized digest capstone-support-v1/sealed_test.jsonl",
      "result": "PASS",
      "actual": "8bf6eb3e153e38ff0b5700cdec21da7bf422bd2b092d945be94eb6f87d68811e",
      "expected": "8bf6eb3e153e38ff0b5700cdec21da7bf422bd2b092d945be94eb6f87d68811e"
    },
    {
      "name": "all SFT target formats capstone-support-v1/sealed_test.jsonl",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "materialized digest capstone-support-v1/train.jsonl",
      "result": "PASS",
      "actual": "8451fd5597d2fa87172a42724b233875b020b1967374c7804a0bab79e1d46cdb",
      "expected": "8451fd5597d2fa87172a42724b233875b020b1967374c7804a0bab79e1d46cdb"
    },
    {
      "name": "all SFT target formats capstone-support-v1/train.jsonl",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "materialized digest capstone-support-v1/validation.jsonl",
      "result": "PASS",
      "actual": "f61ff62c7adc9967421c83411415e044f2df1b5c4ee7b764e89debc4370c3366",
      "expected": "f61ff62c7adc9967421c83411415e044f2df1b5c4ee7b764e89debc4370c3366"
    },
    {
      "name": "all SFT target formats capstone-support-v1/validation.jsonl",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "materialized digest data-clinic-leaky-v1/test.jsonl",
      "result": "PASS",
      "actual": "32f306992f93b72a079f56bff51d4927176b213d6debd3b4edb2416bb564102a",
      "expected": "32f306992f93b72a079f56bff51d4927176b213d6debd3b4edb2416bb564102a"
    },
    {
      "name": "materialized digest data-clinic-leaky-v1/train.jsonl",
      "result": "PASS",
      "actual": "529b7c37eeaba1f169ff434911dbcaa89a270d7fc6750aff9bac637671a7c281",
      "expected": "529b7c37eeaba1f169ff434911dbcaa89a270d7fc6750aff9bac637671a7c281"
    },
    {
      "name": "materialized digest data-clinic-leaky-v1/validation.jsonl",
      "result": "PASS",
      "actual": "154dd58065afe86284620deb850a698b0dac2b4dede9fd040fb50b5cb47d1131",
      "expected": "154dd58065afe86284620deb850a698b0dac2b4dede9fd040fb50b5cb47d1131"
    },
    {
      "name": "materialized digest data-clinic-v1/test.jsonl",
      "result": "PASS",
      "actual": "43450deb84262afa013bbcb754913a17d982fd90901df4df05b8d2b9ea2b463c",
      "expected": "43450deb84262afa013bbcb754913a17d982fd90901df4df05b8d2b9ea2b463c"
    },
    {
      "name": "materialized digest data-clinic-v1/train.jsonl",
      "result": "PASS",
      "actual": "529b7c37eeaba1f169ff434911dbcaa89a270d7fc6750aff9bac637671a7c281",
      "expected": "529b7c37eeaba1f169ff434911dbcaa89a270d7fc6750aff9bac637671a7c281"
    },
    {
      "name": "materialized digest data-clinic-v1/validation.jsonl",
      "result": "PASS",
      "actual": "fefc7e5a5098a298c29319623979d715ef5534374292c1b5c930100e12c34ce9",
      "expected": "fefc7e5a5098a298c29319623979d715ef5534374292c1b5c930100e12c34ce9"
    },
    {
      "name": "materialized digest retrieval-manual-v1/documents.jsonl",
      "result": "PASS",
      "actual": "d6cca516a0c75b7aaa3cb85ca04fea6da06d0837b0592cf43dafcbb292a538aa",
      "expected": "d6cca516a0c75b7aaa3cb85ca04fea6da06d0837b0592cf43dafcbb292a538aa"
    },
    {
      "name": "materialized digest retrieval-manual-v1/queries.jsonl",
      "result": "PASS",
      "actual": "c6e0a692eaccf5427e68fdb231bfcbb6c1dcf7c0c91b9edbcc934e74dccd33f5",
      "expected": "c6e0a692eaccf5427e68fdb231bfcbb6c1dcf7c0c91b9edbcc934e74dccd33f5"
    },
    {
      "name": "pinned template digest",
      "result": "PASS",
      "actual": "872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788",
      "expected": "872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788"
    },
    {
      "name": "model architecture class",
      "result": "PASS",
      "actual": "LlamaForCausalLM",
      "expected": "LlamaForCausalLM"
    },
    {
      "name": "architecture hidden_size",
      "result": "PASS",
      "actual": 576,
      "expected": 576
    },
    {
      "name": "architecture num_hidden_layers",
      "result": "PASS",
      "actual": 30,
      "expected": 30
    },
    {
      "name": "architecture num_attention_heads",
      "result": "PASS",
      "actual": 9,
      "expected": 9
    },
    {
      "name": "architecture num_key_value_heads",
      "result": "PASS",
      "actual": 3,
      "expected": 3
    },
    {
      "name": "architecture max_position_embeddings",
      "result": "PASS",
      "actual": 8192,
      "expected": 8192
    },
    {
      "name": "architecture vocab_size",
      "result": "PASS",
      "actual": 49152,
      "expected": 49152
    },
    {
      "name": "architecture intermediate_size",
      "result": "PASS",
      "actual": 1536,
      "expected": 1536
    },
    {
      "name": "architecture rope_theta",
      "result": "PASS",
      "actual": 100000,
      "expected": 100000
    },
    {
      "name": "architecture tie_word_embeddings",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "download total",
      "result": "PASS",
      "actual": 272437573,
      "expected": 272437573
    },
    {
      "name": "metadata size config.json",
      "result": "PASS",
      "actual": 861,
      "expected": 861
    },
    {
      "name": "metadata digest config.json",
      "result": "PASS",
      "actual": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a",
      "expected": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a"
    },
    {
      "name": "metadata size tokenizer_config.json",
      "result": "PASS",
      "actual": 3764,
      "expected": 3764
    },
    {
      "name": "metadata digest tokenizer_config.json",
      "result": "PASS",
      "actual": "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
      "expected": "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df"
    },
    {
      "name": "light text contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "light muted contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "light action contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "light focus contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "light recorded contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "light simulation contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "light error contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "dark text contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "dark muted contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "dark action contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "dark focus contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "dark recorded contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "dark simulation contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    },
    {
      "name": "dark error contrast threshold",
      "result": "PASS",
      "actual": true,
      "expected": true
    }
  ],
  "retrieval_rankings": [
    {
      "query_id": "RQ01",
      "ranked": [
        {
          "record_id": "RM01",
          "score": 0.5044368928581016
        },
        {
          "record_id": "RM02",
          "score": 0.0
        },
        {
          "record_id": "RM03",
          "score": 0.0
        }
      ]
    },
    {
      "query_id": "RQ02",
      "ranked": [
        {
          "record_id": "RM02",
          "score": 0.34655096811857145
        },
        {
          "record_id": "RM04",
          "score": 0.27720198880720437
        },
        {
          "record_id": "RM07",
          "score": 0.06261552473805203
        }
      ]
    },
    {
      "query_id": "RQ03",
      "ranked": [
        {
          "record_id": "RM03",
          "score": 0.5329571614592222
        },
        {
          "record_id": "RM06",
          "score": 0.11350474029873131
        },
        {
          "record_id": "RM05",
          "score": 0.08247794972081855
        }
      ]
    },
    {
      "query_id": "RQ04",
      "ranked": [
        {
          "record_id": "RM04",
          "score": 0.46350816098461767
        },
        {
          "record_id": "RM01",
          "score": 0.0
        },
        {
          "record_id": "RM02",
          "score": 0.0
        }
      ]
    },
    {
      "query_id": "RQ05",
      "ranked": [
        {
          "record_id": "RM05",
          "score": 0.5910418222222802
        },
        {
          "record_id": "RM02",
          "score": 0.145066531654174
        },
        {
          "record_id": "RM04",
          "score": 0.13797832500708695
        }
      ]
    },
    {
      "query_id": "RQ06",
      "ranked": [
        {
          "record_id": "RM06",
          "score": 0.5619882961489768
        },
        {
          "record_id": "RM04",
          "score": 0.14498942582688443
        },
        {
          "record_id": "RM07",
          "score": 0.10683960606880566
        }
      ]
    }
  ],
  "palette_calculations": [
    {
      "theme": "light",
      "foreground": "text",
      "color": "#24334a",
      "background": "#ffffff",
      "ratio": 12.74715327427825,
      "threshold": 4.5
    },
    {
      "theme": "light",
      "foreground": "muted",
      "color": "#59687b",
      "background": "#ffffff",
      "ratio": 5.689657915623451,
      "threshold": 4.5
    },
    {
      "theme": "light",
      "foreground": "action",
      "color": "#315bae",
      "background": "#ffffff",
      "ratio": 6.485031702776482,
      "threshold": 4.5
    },
    {
      "theme": "light",
      "foreground": "focus",
      "color": "#986400",
      "background": "#ffffff",
      "ratio": 5.050567496040966,
      "threshold": 3
    },
    {
      "theme": "light",
      "foreground": "recorded",
      "color": "#14675e",
      "background": "#ffffff",
      "ratio": 6.706106969348832,
      "threshold": 4.5
    },
    {
      "theme": "light",
      "foreground": "simulation",
      "color": "#805218",
      "background": "#ffffff",
      "ratio": 6.692276293165395,
      "threshold": 4.5
    },
    {
      "theme": "light",
      "foreground": "error",
      "color": "#8a3b12",
      "background": "#ffffff",
      "ratio": 7.73487762982462,
      "threshold": 4.5
    },
    {
      "theme": "dark",
      "foreground": "text",
      "color": "#e5ecf6",
      "background": "#1b283b",
      "ratio": 12.496209240561607,
      "threshold": 4.5
    },
    {
      "theme": "dark",
      "foreground": "muted",
      "color": "#afbdd0",
      "background": "#1b283b",
      "ratio": 7.792300694775018,
      "threshold": 4.5
    },
    {
      "theme": "dark",
      "foreground": "action",
      "color": "#a7c5ff",
      "background": "#1b283b",
      "ratio": 8.543008878131262,
      "threshold": 4.5
    },
    {
      "theme": "dark",
      "foreground": "focus",
      "color": "#edc366",
      "background": "#1b283b",
      "ratio": 8.914636199621283,
      "threshold": 3
    },
    {
      "theme": "dark",
      "foreground": "recorded",
      "color": "#95dec7",
      "background": "#1b283b",
      "ratio": 9.588474692424407,
      "threshold": 4.5
    },
    {
      "theme": "dark",
      "foreground": "simulation",
      "color": "#f0cc8b",
      "background": "#1b283b",
      "ratio": 9.704438457459375,
      "threshold": 4.5
    },
    {
      "theme": "dark",
      "foreground": "error",
      "color": "#ffc49b",
      "background": "#1b283b",
      "ratio": 9.638115346028165,
      "threshold": 4.5
    }
  ],
  "limitations": [
    "No browser rendering or accessibility behavior tested.",
    "No tokenizer/model/dependency runtime installed or executed.",
    "These are independent calculations over authored specification fixtures, not product tests."
  ]
}
