export type Stage = 'detect' | 'analyze' | 'refine' | 'reconstruct' | 'preview' | 'process' | 'export';
export type JobStage = Stage | 'import';
export type CameraRate = [number, number, number, number];
export type ManualFraming = {
  right_to_left_matrix: number[][];
  pre_rate: CameraRate;
  post_rate: CameraRate;
  ease_rate: boolean;
  provenance?: { kind: string; label: string; source_sha256: string; frame: number; calibration_sha256?: string };
};
export type SeamCorrection = {
  geometry: 'auto' | 'off' | 'manual';
  partial_recovery: boolean;
  endpoint_recovery: boolean;
  cadence: boolean;
  rate_easing: boolean;
  color: 'auto' | 'tone' | 'off';
  manual?: ManualFraming;
};
export type SeamResult = {
  frame: number;
  geometry: string;
  geometryReason?: string;
  cadence: boolean;
  rateEasing: boolean;
  color: string;
  notes: string[];
  manual?: ManualFraming;
};
export type Seam = {
  id: string;
  frame: number;
  time: number;
  enabled: boolean;
  origin: 'detected' | 'manual';
  confidence?: number;
  reasons?: string[];
  kind?: string;
  correction?: SeamCorrection;
};
export type AISettings = { configured: boolean; secureStorageAvailable: boolean; provider: 'openai' };
export type SegmentationStatus = { modelId: string; name: string; downloadBytes: number; runtimeAvailable: boolean; downloaded: boolean; available: boolean; reason: string | null; provider: string; license: string; publisher: string };
export type NeuralPrompt = { frame: number; layerId: string; points: { x: number; y: number; label: 1 | 0 }[]; box?: [number, number, number, number] };
export type ReconstructionAction = 'propose' | 'edit' | 'background' | 'render' | 'accept' | 'reject' | 'revert' | 'auto' | 'import' | 'setup-model' | 'segment';
export type MaskStroke = {
  frame: number; layer_id: string; mode: 'include' | 'exclude' | 'protect' | 'emission';
  radius: number; points: [number, number][];
};
export type ReconstructionOptions = {
  reachFrames?: number; motionStrength?: number; allowAI?: boolean; maxAIRequests?: number;
  strokes?: MaskStroke[]; review_approved?: boolean;
  neuralPrompt?: NeuralPrompt;
  segmentation?: 'auto' | 'classic' | 'neural';
};
export type ReconstructionSummary = {
  id: string; manifestPath: string; status: string;
  sourceFrame: number; width: number; height: number; startFrame: number; endFrame: number;
  sourceFramePath: string; candidatePreviewPath?: string;
  previewStartFrame?: number; previewEndFrame?: number;
  layers: { id: string; name: string; role: string; maskPreviewPath: string; keyframes: number[]; confidence?: number }[];
  frames: { frame: number; sourceFramePath: string; masks: Record<string, string> }[];
  issues: string[]; metrics?: Record<string, unknown>; qa?: { autoEligible?: boolean; [key: string]: unknown };
  canRender: boolean; canAccept: boolean; reachFrames?: number | null; motionStrength?: number | null;
};
export type ReconstructionEntry = { candidate?: ReconstructionSummary; accepted?: ReconstructionSummary; history?: ReconstructionSummary[] };
export type Project = {
  version: 1;
  id: string;
  name: string;
  projectPath: string;
  source: string;
  sourceSha256: string;
  metadata: {
    width: number; height: number; fps: number; fps_fraction: string;
    frame_count: number; duration: number; has_audio: boolean;
  };
  seams: Seam[];
  seamResults?: SeamResult[];
  reconstructions?: Record<string, ReconstructionEntry>;
  revision: number;
  refinementBaseline?: { plan: string; planSha256: string; sourceSha256: string; revision: number };
  artifacts: {
    proxy?: string;
    thumbnails?: { frame: number; time: number; path: string }[];
    calibration?: string;
    plan?: string;
    seamPreviews?: { frame: number; path: string; startFrame: number; endFrame: number }[];
    fullPreview?: string;
    export?: string;
    report?: string;
  };
  status: string;
  warnings?: string[];
};
export type JobEvent = {
  jobId: string;
  type: 'progress' | 'complete' | 'error' | 'cancelled';
  stage: JobStage;
  progress?: number;
  message?: string;
  project?: Project;
  error?: string;
};
export type UpdateState = {
  currentVersion: string;
  status: 'disabled' | 'idle' | 'checking' | 'available' | 'downloading' | 'downloaded' | 'installing' | 'error';
  version?: string;
  releaseNotes: string;
  progress?: number;
  checkedAt?: number;
  error?: string;
  busy: boolean;
  message?: string;
};
export interface SeamstressAPI {
  pickVideo(): Promise<string | null>;
  pathForFile(file: File): string;
  createProject(options: { source: string }): Promise<Project>;
  openProject(): Promise<Project | null>;
  getProject(projectPath: string): Promise<Project>;
  setSeams(options: { projectPath: string; seams: Seam[] }): Promise<Project>;
  importSeamCorrection(options: { projectPath: string; frame: number }): Promise<Project | null>;
  importReconstruction(options: { projectPath: string; frame: number }): Promise<Project | null>;
  getAISettings(): Promise<AISettings>;
  setAIKey(key: string): Promise<AISettings>;
  clearAIKey(): Promise<AISettings>;
  getSegmentationStatus(): Promise<SegmentationStatus>;
  run(options: { projectPath: string; stage: Stage; options?: { exportPath?: string; crf?: number; previewSeconds?: number; frame?: number; supportFrames?: number; action?: ReconstructionAction; reconstruction?: ReconstructionOptions; reconstructionEnabled?: boolean } }): Promise<{ jobId: string }>;
  cancelJob(jobId: string): Promise<void>;
  chooseExportPath(options: { suggestedName: string }): Promise<string | null>;
  onJobEvent(callback: (event: JobEvent) => void): () => void;
  mediaUrl(path: string): string;
  revealFile(path: string): Promise<void>;
  getUpdateState(): Promise<UpdateState>;
  checkForUpdates(): Promise<UpdateState>;
  downloadUpdate(): Promise<UpdateState>;
  installUpdate(): Promise<UpdateState>;
  openReleases(): Promise<void>;
  onUpdateState(callback: (state: UpdateState) => void): () => void;
  onOpenUpdates?(callback: () => void): () => void;
}
declare global { interface Window { seamstress?: SeamstressAPI; } }
