import mongoose from "mongoose";

const userSchema = new mongoose.Schema(
  {
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    role: {
      type: String,
      enum: ["doctor", "healthcare_worker", "researcher"],
      default: "doctor",
    },
  },
  { timestamps: true }
);

export default mongoose.model("User", userSchema);
