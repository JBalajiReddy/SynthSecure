import axios from "axios";

const baseURL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000";

export const api = axios.create({
  baseURL,
  headers: { "Content-Type": "application/json" },
  timeout: 15000,
});

export type FeatureInput = number[] | Record<string, any>;
export type PredictPayload = { features: FeatureInput };

export type PredictResponse = {
  prediction: string | string[] | number | number[];
  probability?: number[] | number[][] | null;
  explanations?: Array<{ feature: string; value: number; z: number }>;
};

export async function predictFraud(features: FeatureInput) {
  const { data } = await api.post<PredictResponse>("/predict", {
    features,
  } as PredictPayload);
  return data;
}

export type BaselineResponse = {
  columns: string[];
  stats: Record<string, { mean: number; std: number }>;
};

export async function fetchBaseline() {
  const { data } = await api.get<BaselineResponse>("/metrics/baseline");
  return data;
}

export type FeatureImportanceItem = { feature: string; importance: number };
export type FeatureImportanceResponse = { items: FeatureImportanceItem[] };

export async function fetchFeatureImportance(top: number = 5) {
  const { data } = await api.get<FeatureImportanceResponse>(
    `/metrics/feature-importance?top=${top}`
  );
  return data;
}
