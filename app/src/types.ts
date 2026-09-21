export type Stage = 'detect' | 'analyze' | 'refine' | 'preview' | 'process' | 'export';
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
  run(options: { projectPath: string; stage: Stage; options?: { exportPath?: string; crf?: number; previewSeconds?: number; frame?: number; supportFrames?: number } }): Promise<{ jobId: string }>;
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
